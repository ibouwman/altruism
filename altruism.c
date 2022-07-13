/*
 * altruism.c
 *
 *  Created on: 2 mrt. 2022
 *      Author: Irene Bouwman
 */

//Include
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include "mt64.h"
#include <complex.h>
#include <fftw3.h>
#include "ziggurat_inline.h"

//Declare functions (in order of usage)
//Functions used in main():
void allocateMemory(void);
void createFFTWplans(void);
void createNormalKernel(int, fftw_complex*);
void makeIndividuals(void);
void createLocalDensityMatrix(void);
void fillDensityMatrix(void);
void createExperiencedAltruismMatrix(void);
void fillAltruismMatrix(void);
void moveIndividual(int);
unsigned int positiveModulo(int);
double calculateBirthRate(int);
void considerMutationAndDevelopment(int);
double considerTraitMutation(double, double, double);
double randomExponential(void);
void countPhenotypes(void);
void checkPopulationSize(int);
void updateStates(void);
void destroyFFTWplans(void);
void freeMemory(void);
//Functions used to create output files:
void printRunInfoToFile(FILE*, int);
void printParametersToFile(FILE*);
void printMeanAltruismToFile(FILE*, int);
void printPopulationSizeToFile(FILE*, int);
void printPhenotypesToFile(FILE*, int);
void printTraitsPerIndividualToFile(FILE*);
void printPerCellStatistics(FILE*, int);
void printExperiencedAltruismMatrixToFile(void);
void printDensityMatrixToFile(void);
void printSummedAltruismMatrixToFile(void);
void printSummedPmatrixToFile(void);
void printTraitMatrixToFile(int, int, double, double);
void printSelectionToFile(FILE*, int);
void printSummedMatrixToFile(FILE*, int);
double sumMatrix(fftw_complex*);

//Define parameters and settings, following 2D/parameters in the Fortran code (and Table 1 of the paper)
//Settings
#define TMAX 100001
#define OUTPUTINTERVAL 1250 //Number of timesteps between each output print
#define FIELDS 7 //Number of fields to take into account (in each direction) when creating the normal kernel
#define DELTATIME 0.08 //Multiply rate by DELTATIME to get probability per timestep
#define DELTASPACE 0.1 //Size of a position. This equals 1/resolution in the Fortran code.
#define STEADYSTATEDENSITY (1 - DEATHRATE/BIRTHRATE) * K
#define GRIDSIZE (N * DELTASPACE) * (N * DELTASPACE) //Actual size of the grid, not in terms of DELTASPACE
#define INITIALALTRUISM 0.0
#define INITIALP 1.0 //Initial probability that offspring will have phenotype A
#define THRESHOLD 0.0000000001 //Numbers lower than this are set to 0
#define NBINSALTRUISM 10
#define NBINSP 10
//Parameters
#define BIRTHRATE 5.0 //Baseline max birth rate
#define DEATHRATE 1.0
#define MUTATIONPROBABILITYALTRUISM 0.001
#define MUTATIONPROBABILITYP 0.001
#define MEANMUTSIZEALTRUISM 0.005
#define MEANMUTSIZEP 0.02
#define ALTRUISMSCALE 1
#define COMPETITIONSCALE 4
#define DIFFUSIONCONSTANT 0.04
#define MOVEMENTSCALE sqrt(2 * DIFFUSIONCONSTANT * DELTATIME)
#define K 40 //Carrying capacity
#define B0 1.0 //Basal benefit of altruism
#define BMAX 5.0 //Maximum benefit of altruism
#define ALPHA 0.75 //Use ALPHA = 1 for the original model
#define BETA (1 - ALPHA)*KAPPA
#define KAPPA 0.1
#define N 1024 //2**9
#define NPOS N * N

//Declare structures
struct Individual {
	int xpos;
	int ypos;
	double altruism;
	double p;
	int phenotype; //altruism expressed (1) or not expressed (0)
	int offspring;
};

//Declare global variables
int population_size_old;
int population_size_new;
fftw_complex* normal_for_density;
fftw_complex* normal_for_density_forward;
fftw_complex* normal_for_altruism;
fftw_complex* normal_for_altruism_forward;
fftw_complex* density;
fftw_complex* density_forward;
fftw_complex* normal_forward_density_forward_product;
fftw_complex* normal_density_convolution;
fftw_complex* altruism;
fftw_complex* altruism_forward;
fftw_complex* normal_forward_altruism_forward_product;
fftw_complex* normal_altruism_convolution;

//fftw_plan names correspond to the fftw_complex objects their output is stored in
fftw_plan fftw_plan_normal_for_density_forward;
fftw_plan fftw_plan_normal_for_altruism_forward;
fftw_plan fftw_plan_density_forward;
fftw_plan fftw_plan_normal_density_convolution;
fftw_plan fftw_plan_altruism_forward;
fftw_plan fftw_plan_normal_altruism_convolution;

struct Individual* individuals_old; //This is the 'old state'
struct Individual* individuals_new; //This is the 'new state'
struct Individual** individuals_old_ptr;
struct Individual** individuals_new_ptr;
int* density_A;
int* density_B;
double* sumaltr; //Matrix for summed altruism of all individuals. Summed altruism of all As is in the altruism fftw complex object
double* sumaltr_B;
double* sump;
double* sump_A;
double* sump_B;
int* trait_matrix;

int INITIALPOPULATIONSIZE = round(STEADYSTATEDENSITY * GRIDSIZE); //Initial and maximal population size depend on steady state density and grid size.
int MAXPOPULATIONSIZE = round(1.5 * K * GRIDSIZE); //Note that MAXPOPULATIONSIZE can be larger than NPOS because multiple individuals are allowed at the same position.
int newborns;
int deaths;
int A_counter;
int B_counter;

//Output files:
FILE *expaltr_file;
FILE *density_file;
FILE *density_file_A;
FILE *density_file_B;
FILE *sumaltr_file;
FILE *sumaltr_file_A;
FILE *sumaltr_file_B;
FILE *sump_file;
FILE *sump_file_A;
FILE *sump_file_B;
FILE *runinfo_file;
FILE *trait_matrix_file;
FILE *selection_file;
char filename_experienced_altruism[50];
char filename_density[50];;
char filename_density_A[50];;
char filename_density_B[50];
char filename_summed_altruism[50];
char filename_summed_altruism_A[50];
char filename_summed_altruism_B[50];
char filename_summed_p[50];
char filename_summed_p_A[50];
char filename_summed_p_B[50];
char filename_trait_matrix[50];
char filename_runinfo[50];
char filename_selection[50];
char run_id[] = "00000"; //Give your run a unique id to prevent overwriting of output files

//Main
int main() {
	clock_t start = clock();
	time_t tm;
	time(&tm);
	printf("Running %s phenotypic-differentiation branch. Started at %s\n", __FILE__, ctime(&tm));
	init_genrand64(1); //Use time(0) for different numbers every run (not reproducible!)
	uint32_t jsr_value = 123456789; //Values taken from ziggurat_inline_test.c (available online)
	uint32_t jcong_value = 234567891;
	uint32_t w_value = 345678912;
	uint32_t z_value = 456789123;
	zigset(jsr_value, jcong_value, w_value, z_value);
	allocateMemory();
	createFFTWplans();
	printf("Creating kernels...\n");
	createNormalKernel(COMPETITIONSCALE, normal_for_density); //Create and execute only once, same for each timestep
	createNormalKernel(ALTRUISMSCALE, normal_for_altruism);
	fftw_execute(fftw_plan_normal_for_density_forward);
	fftw_execute(fftw_plan_normal_for_altruism_forward);
	printf("Creating individuals...\n");
	makeIndividuals();
	population_size_old = INITIALPOPULATIONSIZE;
	population_size_new = 0;
	int counter = 1; //Counter for file naming
	sprintf(filename_runinfo, "%s_runinfo.txt", run_id);
	runinfo_file = fopen(filename_runinfo, "w+");
	sprintf(filename_selection, "%s_selection.txt", run_id);
	selection_file = fopen(filename_selection, "w+");
    for (int t = 0; t < TMAX; t++) {
    	if(t == 0){
    		printf("Simulation has started!\nProgress (printed every 5 timesteps):\n");
    	}
    	countPhenotypes();
    	if((A_counter + B_counter) != population_size_old){
    		printf("ERROR: The summed number of individuals per phenotype (A: %d, B: %d) doesn't equal the population size (%d)!\n", A_counter, B_counter, population_size_old);
    		exit(1);
    	}
    	printRunInfoToFile(runinfo_file, t);
    	newborns = 0;
		deaths = 0;
    	createLocalDensityMatrix();
    	createExperiencedAltruismMatrix();
    	if(t % 5 == 0){
    		printf("%d out of %d timesteps.\n", t, TMAX);
    	}
    	if(t % OUTPUTINTERVAL == 0){
        	sprintf(filename_experienced_altruism, "%s_expaltr_%04d.txt", run_id, counter);
        	expaltr_file = fopen(filename_experienced_altruism, "w+");
        	sprintf(filename_density, "%s_density_%04d.txt", run_id, counter);
        	density_file = fopen(filename_density, "w+");
        	sprintf(filename_density_A, "%s_densityA_%04d.txt", run_id, counter);
        	density_file_A = fopen(filename_density_A, "w+");
        	sprintf(filename_density_B, "%s_densityB_%04d.txt", run_id, counter);
        	density_file_B = fopen(filename_density_B, "w+");
        	sprintf(filename_summed_altruism, "%s_sumaltr_%04d.txt", run_id, counter);
        	sumaltr_file = fopen(filename_summed_altruism, "w+");
        	sprintf(filename_summed_altruism_A, "%s_sumaltrA_%04d.txt", run_id, counter);
        	sumaltr_file_A = fopen(filename_summed_altruism_A, "w+");
        	sprintf(filename_summed_altruism_B, "%s_sumaltrB_%04d.txt", run_id, counter);
        	sumaltr_file_B = fopen(filename_summed_altruism_B, "w+");
        	sprintf(filename_summed_p, "%s_sump_%04d.txt", run_id, counter);
        	sump_file = fopen(filename_summed_p, "w+");
        	sprintf(filename_summed_p_A, "%s_sumpA_%04d.txt", run_id, counter);
        	sump_file_A = fopen(filename_summed_p_A, "w+");
        	sprintf(filename_summed_p_B, "%s_sumpB_%04d.txt", run_id, counter);
        	sump_file_B = fopen(filename_summed_p_B, "w+");
        	sprintf(filename_trait_matrix, "%s_traitmatrix_%04d.txt", run_id, counter);
        	trait_matrix_file = fopen(filename_trait_matrix, "w+");
        	counter++;
        	printExperiencedAltruismMatrixToFile();
        	printDensityMatrixToFile();
        	printSummedAltruismMatrixToFile();
        	printSummedPmatrixToFile();
        	printTraitMatrixToFile(NBINSALTRUISM, NBINSP, 0.01, 0.1); //TODO: Think about global/local variables, or adding maxvalue for p or altruism instead of binsize
    	}
		for (int i = 0; i < population_size_old; i++){
			int i_new = i + newborns - deaths; //The index of i in the new timestep, taking into account births and deaths the current timestep
			if(individuals_new[i_new].offspring != 0){
				printf("Error! Individual in the new state can't haven non-zero offspring.\n"); //TODO: This error message can be removed once sure that offspring is working correctly
				exit(0);
			}
			double probabilityOfEvent = genrand64_real2();
			if (probabilityOfEvent < DEATHRATE*DELTATIME){ //If individual dies...
				deaths += 1;
			}
			else{ //If individual doesn't die...
				//First: Administration:
				individuals_new[i_new] = individuals_old[i];
				individuals_old[i].offspring += 1; //Update offspring for the 'old' state, to calculate selection
				population_size_new += 1;
				double birth_rate = calculateBirthRate(i_new);
				//Second: Action:
				moveIndividual(i_new); //...Move it
				if (probabilityOfEvent < DEATHRATE*DELTATIME + birth_rate*DELTATIME){ //If the individual reproduces...
					//First: Administration:
					individuals_new[i_new + 1] = individuals_old[i]; //Initially child = parent
					individuals_new[i_new + 1].offspring = 0; //But with offspring 0
					population_size_new += 1; //Add child to the population size of the next timestep
					newborns += 1; //Add child to the newborns of this timestep
					individuals_old[i].offspring += 1; //Update offspring of the parent for the 'old' state
					//Second: Action:
					considerMutationAndDevelopment(i_new + 1); //Consider trait mutation and phenotype development of the child
					moveIndividual(i_new + 1); //Move the child
				}
			}
		}
    	checkPopulationSize(t);
    	if(t % OUTPUTINTERVAL == 0){
        	printSelectionToFile(selection_file, t);
    	}
    	updateStates(); //New state becomes old state
   }
   destroyFFTWplans();
   freeMemory();
   clock_t end = clock();
   double runtime = (double)((end - start)/CLOCKS_PER_SEC);
   printf("\nDone! Run took %f seconds.\n", runtime);
   return 0;
}

/**
 * Allocates memory for the arrays and fftw_complex objects used in the code.
 */
void allocateMemory(void){
    individuals_old = malloc(MAXPOPULATIONSIZE * sizeof(struct Individual));
    individuals_new = malloc(MAXPOPULATIONSIZE * sizeof(struct Individual));
    if (individuals_old == NULL) {
        printf("ERROR: Memory for individuals_old not allocated.\n");
        exit(1);
    }
    else if (individuals_new == NULL) {
        printf("ERROR: Memory for individuals_new not allocated.\n");
        exit(1);
    }
    density_A = malloc(NPOS * sizeof(int));
    density_B = malloc(NPOS * sizeof(int));
    sumaltr = malloc(NPOS * sizeof(double));
    sumaltr_B = malloc(NPOS * sizeof(double));
    sump = malloc(NPOS * sizeof(double));
    sump_A = malloc(NPOS * sizeof(double));
    sump_B = malloc(NPOS * sizeof(double));
    trait_matrix = malloc(NBINSALTRUISM * NBINSP * sizeof(int));
	normal_for_density = fftw_alloc_complex(NPOS);
	normal_for_altruism = fftw_alloc_complex(NPOS);
    normal_for_density_forward = fftw_alloc_complex(NPOS);
    normal_for_altruism_forward = fftw_alloc_complex(NPOS);
    density = fftw_alloc_complex(NPOS);
    density_forward = fftw_alloc_complex(NPOS);
    normal_forward_density_forward_product = fftw_alloc_complex(NPOS);
    normal_density_convolution = fftw_alloc_complex(NPOS);
    altruism = fftw_alloc_complex(NPOS);
    altruism_forward = fftw_alloc_complex(NPOS);
    normal_forward_altruism_forward_product = fftw_alloc_complex(NPOS);
    normal_altruism_convolution = fftw_alloc_complex(NPOS);
}

/**
 * Creates plans for the Fourier transformations used in the code.
 */
void createFFTWplans(void){
	fftw_plan_normal_for_density_forward = fftw_plan_dft_2d(N, N, normal_for_density, normal_for_density_forward, FFTW_FORWARD, FFTW_MEASURE);
	fftw_plan_normal_for_altruism_forward = fftw_plan_dft_2d(N, N, normal_for_altruism, normal_for_altruism_forward, FFTW_FORWARD, FFTW_MEASURE);
	fftw_plan_density_forward = fftw_plan_dft_2d(N, N, density, density_forward, FFTW_FORWARD, FFTW_MEASURE);
	fftw_plan_normal_density_convolution = fftw_plan_dft_2d(N, N, normal_forward_density_forward_product, normal_density_convolution, FFTW_BACKWARD, FFTW_MEASURE);
	fftw_plan_altruism_forward = fftw_plan_dft_2d(N, N, altruism, altruism_forward, FFTW_FORWARD, FFTW_MEASURE);
	fftw_plan_normal_altruism_convolution = fftw_plan_dft_2d(N, N, normal_forward_altruism_forward_product, normal_altruism_convolution, FFTW_BACKWARD, FFTW_MEASURE);
}


/**
 * Creates a normal kernel. Numbers lower than THRESHOLD are set to 0. This should equal line 34-61 in the Fortran kernels.f90 file.
 */
void createNormalKernel(int scale, fftw_complex* normal_kernel2D){
	fftw_complex* normal_kernel1D;
	normal_kernel1D = fftw_alloc_complex(N);
	double preFactor = 1.0/(2.0*(scale*(1/DELTASPACE))*(scale*(1/DELTASPACE)));
	for(int x = 0; x < N; x++){
		double exp_counter = 0.0;
		for(int field = -FIELDS; field < FIELDS; field++){
			exp_counter += exp(-preFactor * ((x + field*N)*(x + field*N))); //TODO: Change summation strategy so small numbers don't disappear
		}
		if(exp_counter < THRESHOLD){
			exp_counter = 0;
		}
		normal_kernel1D[x] = creal(exp_counter); //Fortran uses partial summation, which is slightly different
	}
	int index = 0;
	double kernel_sum = 0.0;
	for(int x = 0; x < N; x++){
		for(int y = 0; y < N; y++){
			normal_kernel2D[index] = normal_kernel1D[x]*normal_kernel1D[y];
			kernel_sum += normal_kernel2D[index];
			index++;
		}
	}
	for(int i = 0; i < NPOS; i++){
		normal_kernel2D[i] = (1/DELTASPACE)*(1/DELTASPACE) * normal_kernel2D[i]/kernel_sum;
	}
	fftw_free(normal_kernel1D);
}

/**
 * Creates the initial individuals. Called once at the beginning of the code.
 */
void makeIndividuals(){
	for (int i = 0; i < INITIALPOPULATIONSIZE; i++){
		double random_x = genrand64_real2();
		individuals_old[i].xpos = ceil(random_x * N); //ceil so individuals can't have position 0
		double random_y = genrand64_real2();
		individuals_old[i].ypos = ceil(random_y * N);
		individuals_old[i].altruism = INITIALALTRUISM;
		individuals_old[i].p = INITIALP;
		individuals_old[i].phenotype = 1; //Initially, all individuals are A
		individuals_old[i].offspring = 0;
	}
}

/**
 * Creates a matrix with the local density at each position in the field using convolution of the normal kernel created above and the density matrix created in fillDensityMatrix().
 * //TODO: Create a convolution() function with two input matrices (altruism or density, and kernel) and two input plans, that can be used instead of createLocalDensityMatrix and createExperiencedAltruismMatrix
 */
void createLocalDensityMatrix(){
	fillDensityMatrix();
	fftw_execute(fftw_plan_density_forward); //Forward Fourier of density matrix
	for (int position = 0; position < NPOS; position++){ //Multiply forward-transformed kernel and density matrices
	    normal_forward_density_forward_product[position] = normal_for_density_forward[position] * density_forward[position];
	}
	fftw_execute(fftw_plan_normal_density_convolution); //Convolution: Inverse Fourier of product of forward-transformed kernel and density matrices
	for (int element = 0; element < NPOS; element++){
		normal_density_convolution[element] = normal_density_convolution[element] / creal(NPOS); //Output of FFTW_BACKWARD is automatically multiplied by number of elements, so divide by number of elements (NPOS) to get local density
	}
}

/**
 * Creates a matrix with the absolute number of individuals at each position in the field.
 * //TODO: Largely the same as fillAltruismMatrix(), it's probably more elegant to make one function for both
 */
void fillDensityMatrix(){
	for (int i = 0; i < population_size_old; i++){
		int position_of_individual = (individuals_old[i].xpos - 1) * N + (individuals_old[i].ypos - 1);
		density[position_of_individual] += 1;
		if(individuals_old[i].phenotype == 1){
			density_A[position_of_individual] += 1;
		}else{
			density_B[position_of_individual] += 1;
		}
	}
}

/**
 * Creates a matrix with the experienced level of altruism at each position using convolution of the normal kernel and the altruism matrix created in fillAltruismMatrix().
 * Similar to the creation of the local density matrix.
 */
void createExperiencedAltruismMatrix(){
	fillAltruismMatrix();
	fftw_execute(fftw_plan_altruism_forward); //Forward Fourier of the altruism matrix
	for (int position = 0; position < NPOS; position++){
		normal_forward_altruism_forward_product[position] = normal_for_altruism_forward[position] * altruism_forward[position];
	}
	fftw_execute(fftw_plan_normal_altruism_convolution); //Convolution
	for (int element = 0; element < NPOS; element++){
		normal_altruism_convolution[element] = normal_altruism_convolution[element] / creal(NPOS);
	}
}

/**
 * Creates a matrix with for each position in the field the cumulative level of EXPRESSED altruism of the individuals at that position,
 * and stores it in an fftw complex object called altruism. Also stores summed p and altruism levels in matrices for output.
 * Similar to the creation of the density matrix.
 */
void fillAltruismMatrix(){
	for (int i = 0; i < population_size_old; i++){
		int position_of_individual = (individuals_old[i].xpos - 1) * N + (individuals_old[i].ypos - 1);
		sumaltr[position_of_individual] += individuals_old[i].altruism;
		sump[position_of_individual] += individuals_old[i].p;
		if (individuals_old[i].phenotype == 1){
			altruism[position_of_individual] += individuals_old[i].altruism;
			sump_A[position_of_individual] += individuals_old[i].p;
		} else {
			sumaltr_B[position_of_individual] += individuals_old[i].altruism;
			sump_B[position_of_individual] += individuals_old[i].p;
		}
	}
}

/**
 * Assigns a new position in the field to the input individual.
 * index_of_individual: The index of the individual in the individuals_new array. Make sure that the individual has been copied to the new state before this function is called.
 */
void moveIndividual(int index_of_individual){ //TODO: Look into more efficient method: only use modulo operation if new position is out of bounds?
	float normal_x = r4_nor_value();
	float normal_y = r4_nor_value();
	int move_x = round((MOVEMENTSCALE/DELTASPACE) * normal_x);
	int move_y = round((MOVEMENTSCALE/DELTASPACE) * normal_y);
	individuals_new[index_of_individual].xpos = positiveModulo((individuals_new[index_of_individual].xpos + move_x - 1)) + 1;
	individuals_new[index_of_individual].ypos = positiveModulo((individuals_new[index_of_individual].ypos + move_y - 1)) + 1;
}

/**
 * Calculates an always positive modulo with divisor N. Note that N = N so the function can also be used for movement in the y direction.
 * dividend: The dividend of the modulo operation.
 * returns: dividend % N, where the result is always positive.
 */
unsigned int positiveModulo(int dividend){
	int modulo = dividend % N;
	if(modulo < 0){
		modulo += N;
	}
	return modulo;
}

/**
 * Calculates the birth rate of the input individual.
 * i: The individual whose birth rate is calculated.
 * returns: The birth rate of the individual.
 */
double calculateBirthRate(int index_of_parent){
	int position = (individuals_new[index_of_parent].xpos - 1) * N + (individuals_new[index_of_parent].ypos - 1); //Convert x and y coordinates of individual to find corresponding position in fftw_complex object
	double local_density = normal_density_convolution[position];
	double experienced_altruism = normal_altruism_convolution[position];
	double benefit = (BMAX * experienced_altruism)/((BMAX/B0) + experienced_altruism);
	double cost =  ALPHA*individuals_new[index_of_parent].altruism + (BETA*individuals_new[index_of_parent].altruism)/(KAPPA + individuals_new[index_of_parent].altruism);
	double effectiveCost = individuals_new[index_of_parent].phenotype * cost; //Only individuals that express altruism (phenotype = 1) pay a cost
	double birth_rate = BIRTHRATE * (1.0 - effectiveCost + benefit) * (1.0 - (local_density/K));
	if (birth_rate < 0){
		birth_rate = 0; //Negative birth rates are set to 0
	}
	return birth_rate;
}

/**
 * Considers mutation and development of the child individual.
 * index_of_child: The index of the child in the individuals_new array.
 */
void considerMutationAndDevelopment(int index_of_child){
	individuals_new[index_of_child].altruism = considerTraitMutation(individuals_new[index_of_child].altruism, MEANMUTSIZEALTRUISM, MUTATIONPROBABILITYALTRUISM);
	individuals_new[index_of_child].p = considerTraitMutation(individuals_new[index_of_child].p, MEANMUTSIZEP, MUTATIONPROBABILITYP);
	if(individuals_new[index_of_child].p > 1.0){ //Probability cannot become larger than 1
		individuals_new[index_of_child].p = 1.0;
	} //TODO: Warning for altruism > 1?
	double random_phenotype = genrand64_real2(); //Is altruism expressed or not? Depends on p
	if (random_phenotype < individuals_new[index_of_child].p){ //p is the probability to express altruism
		individuals_new[index_of_child].phenotype = 1;
	}
	else{
		individuals_new[index_of_child].phenotype = 0;
	}
}

/**
 * Considers whether mutation occurs. If so, the trait value is changed.
 * trait: The trait under consideration.
 * mean_mutation_size: The mean size of a mutation in the trait.
 * return: The value of the trait after considering mutation.
 */
double considerTraitMutation(double trait, double mean_mutation_size, double mutation_probability){
	double random_number = genrand64_real2();
	if(random_number < mutation_probability){
		double delta_trait;
		double random_direction = genrand64_real2(); //...Randomly decide direction of mutation
		if(random_direction < 0.5){
			delta_trait = -mean_mutation_size * randomExponential(); //...Calculate change in trait value due to mutation
		}
		else{
			delta_trait = mean_mutation_size * randomExponential(); //...Calculate new trait value (trait value of child initially equals trait value of parent)
		}
		trait = trait + delta_trait;
		if(trait < 0){
			trait = 0;
		}
	}
	return trait;
}

/**
 * Draws a random number from an exponential distribution.
 */
double randomExponential(void){
	double random = genrand64_real2();
	double random_exponential = -log(random);
	return random_exponential;
}

/**
 * Counts the number of individuals per phenotype. Should be called at the beginning of a timestep.
 * NOTE that A_counter and B_counter are only accurate if this function has been called recently!
 */
void countPhenotypes(void){
	A_counter = 0;
	B_counter = 0;
	for(int i = 0; i < population_size_old; i++){
		if(individuals_old[i].phenotype == 1){
			A_counter += 1;
		}
		else{
			B_counter += 1;
		}
	}
}

/**
 * Checks whether the population size is not out of bounds.
 * Stops run and throws error when population size is above MAXPOPULATIONSIZE or below 0.
 * Stops run and prints message when population size is 0 i.e. population died out.
 */
void checkPopulationSize(int t){
	if ((population_size_new > MAXPOPULATIONSIZE) || population_size_new < 0){
		printf("\nERROR: Population size must be between 0 and %d, but population size for next timestep (t = %d) is %d.\n", MAXPOPULATIONSIZE, t+1, population_size_new);
		exit(1);
	}
	else if (population_size_new == 0){
		printf("\n%d individuals left in next timestep. Population died out at t = %d!\n", population_size_new, t);
		exit(1);
	}
}

/**
 * Updates the old and new state for the next timestep, i.e. old individuals are replaced by new individuals. Resets all pointers with timestep-specific information.
 */
void updateStates(){
	individuals_old_ptr = &individuals_old; //Make pointers to pointers, this is necessary to swap old and new individuals
	individuals_new_ptr = &individuals_new;
	memset(individuals_old, 0, MAXPOPULATIONSIZE * sizeof(*individuals_old)); //Delete the old individuals, i.e. set array to 0
	struct Individual* temp = *individuals_old_ptr; //Make a temp ptr that points to the old individuals, now filled with 0s
	*individuals_old_ptr = *individuals_new_ptr; //New individuals become old individuals
	*individuals_new_ptr = temp; //Reset the new individuals for the next state by pointing to the block with 0s
	population_size_old = population_size_new; //Switch to new state, including all individuals that were born or didn't die in the previous timestep
	population_size_new = 0; //Reset value of new state population size
	memset(density_A, 0, NPOS * sizeof(*density_A));
	memset(density_B, 0, NPOS * sizeof(*density_B));
	memset(sumaltr, 0, NPOS * sizeof(*sumaltr));
	memset(sumaltr_B, 0, NPOS * sizeof(*sumaltr_B));
	memset(sump, 0, NPOS * sizeof(*sump));
	memset(sump_A, 0, NPOS * sizeof(*sump_A));
	memset(sump_B, 0, NPOS * sizeof(*sump_B));
	memset(trait_matrix, 0, NBINSALTRUISM * NBINSP * sizeof(*trait_matrix));
	memset(density, 0, NPOS * sizeof(*density));
	memset(density_forward, 0, NPOS * sizeof(*density_forward));
	memset(normal_forward_density_forward_product, 0, NPOS * sizeof(*normal_forward_density_forward_product));
	memset(normal_density_convolution, 0, NPOS * sizeof(*normal_density_convolution));
	memset(altruism, 0, NPOS * sizeof(*altruism));
	memset(altruism_forward, 0, NPOS * sizeof(*altruism_forward));
	memset(normal_forward_altruism_forward_product, 0, NPOS * sizeof(*normal_forward_altruism_forward_product));
	memset(normal_altruism_convolution, 0, NPOS * sizeof(*normal_altruism_convolution));
}

/**
 * Destroys all plans created for the Fourier transformations in createFFTWplans().
 */
void destroyFFTWplans(void){
	fftw_destroy_plan(fftw_plan_normal_for_density_forward);
	fftw_destroy_plan(fftw_plan_normal_for_altruism_forward);
	fftw_destroy_plan(fftw_plan_density_forward);
	fftw_destroy_plan(fftw_plan_normal_density_convolution);
	fftw_destroy_plan(fftw_plan_altruism_forward);
	fftw_destroy_plan(fftw_plan_normal_altruism_convolution);
}

/**
 * Frees all memory allocated for the arrays and fftw_complex objects in allocateMemory().
 */
void freeMemory(void){
	free(individuals_old);
	free(individuals_new);
	free(density_A);
	free(density_B);
	free(sumaltr);
	free(sumaltr_B);
	free(sump);
	free(sump_A);
	free(sump_B);
	free(trait_matrix);
	fftw_free(normal_for_density);
	fftw_free(normal_for_altruism);
	fftw_free(normal_for_density_forward);
	fftw_free(normal_for_altruism_forward);
	fftw_free(density);
	fftw_free(density_forward);
	fftw_free(normal_forward_density_forward_product);
	fftw_free(normal_density_convolution);
	fftw_free(altruism);
	fftw_free(altruism_forward);
	fftw_free(normal_forward_altruism_forward_product);
	fftw_free(normal_altruism_convolution);
}

/**
 * Generates a file with basic information about the simulation. Make sure to call countPhenotypes() first!
 */
void printRunInfoToFile(FILE *filename, int timestep){
	if(timestep == 0){
		printParametersToFile(filename);
		fprintf(filename, "Timestep Time Population_size As Bs  Mean_altruism_all Mean_altruism_A Mean_altruism_B Mean_p_all Mean_p_A Mean_p_B\n");
	}
	if(timestep % OUTPUTINTERVAL == 0){
		double cumulative_altruism = 0; double cumulative_p = 0;
		double cumulative_altruism_A = 0; double cumulative_p_A = 0;
		double cumulative_altruism_B = 0; double cumulative_p_B = 0;
		for(int i = 0; i < population_size_old; i++){
			cumulative_altruism += individuals_old[i].altruism;
			cumulative_p += individuals_old[i].p;
			if(individuals_old[i].phenotype == 1){
				cumulative_altruism_A += individuals_old[i].altruism;
				cumulative_p_A += individuals_old[i].p;
			}else{
				cumulative_altruism_B += individuals_old[i].altruism;
				cumulative_p_B += individuals_old[i].p;
			}
		}
		double mean_altruism = cumulative_altruism/population_size_old; double mean_p = cumulative_p/population_size_old;
		double mean_altruism_A = cumulative_altruism_A/A_counter; double mean_p_A = cumulative_p_A/A_counter;
		double mean_altruism_B = cumulative_altruism_B/B_counter; double mean_p_B = cumulative_p_B/B_counter;
		fprintf(filename, "%d %f %d %d %d %f %f %f %f %f %f\n", timestep, timestep*DELTATIME, population_size_old, A_counter, B_counter, mean_altruism, mean_altruism_A, mean_altruism_B, mean_p, mean_p_A, mean_p_B);
	}
}

/**
 * Print selection to file
 */
void printSelectionToFile(FILE *filename, int timestep){
	if(timestep == 0){
		fprintf(filename, "Timestep Time SelectionP SelectionAltruism\n");
	}
	int cumulative_offspring = 0;
	double cumulative_p = 0;
	double cumulative_altruism = 0;
	for(int i = 0; i < population_size_old; i++){
		cumulative_offspring += individuals_old[i].offspring;
		cumulative_p += individuals_old[i].p;
		cumulative_altruism += individuals_old[i].altruism;
	}
	double cumulative_fitness = 0;
	for(int i = 0; i < population_size_old; i++){
		cumulative_fitness += individuals_old[i].offspring * (population_size_old/cumulative_offspring);
	}
	double mean_fitness = cumulative_fitness/population_size_old;
	double mean_p = cumulative_p/population_size_old;
	double mean_altruism = cumulative_altruism/population_size_old;
	double numerator_p = 0;
	double numerator_altruism = 0;
	for(int i = 0; i < population_size_old; i++){
		double fitness = individuals_old[i].offspring * (population_size_old/cumulative_offspring);
		numerator_p += (fitness - mean_fitness)*(individuals_old[i].p - mean_p);
		numerator_altruism += (fitness - mean_fitness)*(individuals_old[i].altruism - mean_altruism);
	}
	double selection_on_p = numerator_p/population_size_old;
	double selection_on_altruism = numerator_altruism/population_size_old;
	fprintf(filename, "%d %f %f %f\n", timestep, timestep*DELTATIME, selection_on_p, selection_on_altruism);
}

/**
 * Prints predefined parameters to the input file.
 */
void printParametersToFile(FILE *filename){
	time_t tm;
	time(&tm);
	fprintf(filename, "Simulation from %s performed at %s\n", __FILE__, ctime(&tm));
	fprintf(filename, "Output files are created with run id: %s\n", run_id);
	fprintf(filename, "Predefined parameters:\n");
	fprintf(filename, "Tmax = %d\n", TMAX);
	fprintf(filename, "DeltaTime = %f\n", DELTATIME);
	fprintf(filename, "DeltaSpace = %f\n", DELTASPACE);
	fprintf(filename, "Initial population size = %d\n", INITIALPOPULATIONSIZE);
	fprintf(filename, "Number of positions = %d\n", NPOS);
	fprintf(filename, "Initial altruism level = %f\n", INITIALALTRUISM);
	fprintf(filename, "Initial p = %f\n", INITIALP);
	fprintf(filename, "Death rate = %f\n", DEATHRATE);
	fprintf(filename, "Birth rate = %f\n", BIRTHRATE);
	fprintf(filename, "Mutation probability altruism = %f\n", MUTATIONPROBABILITYALTRUISM);
	fprintf(filename, "Mean mutation size altruism = %f\n", MEANMUTSIZEALTRUISM);
	fprintf(filename, "Mutation probability p = %f\n", MUTATIONPROBABILITYP);
	fprintf(filename, "Mean mutation size p = %f\n", MEANMUTSIZEP);
	fprintf(filename, "Scale of altruism = %d\n", ALTRUISMSCALE);
	fprintf(filename, "Scale of competition = %d\n", COMPETITIONSCALE);
	fprintf(filename, "Scale of movement = %f\n", MOVEMENTSCALE);
	fprintf(filename, "B0 = %f\n", B0);
	fprintf(filename, "BMAX = %f\n", BMAX);
	fprintf(filename, "ALPHA = %f\n", ALPHA);
	fprintf(filename, "BETA = %f\n", BETA);
	fprintf(filename, "KAPPA = %f\n", KAPPA);
	fprintf(filename, "K = %d\n", K);
	fprintf(filename, "Number of fields (kernel) = %d\n", FIELDS);
}

/**
 * Calculates the mean level of altruism in the population and prints this to an outputfile. Call countPhenotypes() first!
 * Calls the printParametersToFile() function.
 */
void printMeanAltruismToFile(FILE *filename, int timestep){
	if(timestep == 0){
		printParametersToFile(filename);
		fprintf(filename, "Timestep Time Population_size As Bs Mean_altruism_all Mean_altruism_A Mean_altruism_B Mean_cost_A Mean_effective_cost_A Mean_cost_B Mean_effective_cost_B\n");
	}
	if(timestep % OUTPUTINTERVAL == 0){
		double cumulative_altruism_all = 0.0;
		double cumulative_altruism_A = 0.0;
		double cumulative_altruism_B = 0.0;
		double cumulative_cost_A = 0.0;
		double cumulative_effective_cost_A = 0.0;
		double cumulative_cost_B = 0.0;
		double cumulative_effective_cost_B = 0.0;
		for(int i = 0; i < population_size_old; i++){
			cumulative_altruism_all += individuals_old[i].altruism;
			double cost = ALPHA*individuals_old[i].altruism + (BETA*individuals_old[i].altruism)/(KAPPA + individuals_old[i].altruism);
			double effective_cost = individuals_old[i].phenotype * cost;
			if(individuals_old[i].phenotype == 1){
				cumulative_altruism_A += individuals_old[i].altruism;
				cumulative_cost_A += cost;
				cumulative_effective_cost_A += effective_cost;
			}
			else{
				cumulative_altruism_B += individuals_old[i].altruism;
				cumulative_cost_B += cost;
				cumulative_effective_cost_B += effective_cost;
			}
		}
		double mean_altruism_all = cumulative_altruism_all/population_size_old;
		double mean_altruism_A; double mean_cost_A; double mean_effective_cost_A;
		double mean_altruism_B; double mean_cost_B; double mean_effective_cost_B;
		if(A_counter > 0){
			mean_altruism_A = cumulative_altruism_A/A_counter;
			mean_cost_A = cumulative_cost_A/A_counter;
			mean_effective_cost_A = cumulative_effective_cost_A/A_counter;
		}
		if(B_counter > 0){
			mean_altruism_B = cumulative_altruism_B/B_counter;
			mean_cost_B = cumulative_cost_B/B_counter;
			mean_effective_cost_B = cumulative_effective_cost_B/B_counter;
		}
		fprintf(filename, "%d %f %d %d %d %f %f %f %f %f %f %f\n", timestep, timestep*DELTATIME, population_size_old, A_counter, B_counter, mean_altruism_all, mean_altruism_A, mean_altruism_B, mean_cost_A, mean_effective_cost_A, mean_cost_B, mean_effective_cost_B);
	}
}

/**
 * Prints the population size to an outputfile.
 * Calls the printParametersToFile() function.
 */
void printPopulationSizeToFile(FILE *filename, int timestep){
	if(timestep == 0){
		printParametersToFile(filename);
		fprintf(filename, "Timestep Time Population_size\n");
	}
	if(timestep % OUTPUTINTERVAL == 0){
		fprintf(filename, "%d %f %d\n", timestep, timestep*DELTATIME, population_size_old);
	}
}

/**
 * Prints number of individuals per timestep with their phenotype.
 * Calls the printParametersToFile() function.
 */
void printPhenotypesToFile(FILE *filename, int timestep){
	if(timestep == 0){
		printParametersToFile(filename);
		fprintf(filename, "Timestep Time Population_size As Bs\n");
	}
	if(timestep % OUTPUTINTERVAL == 0){
		fprintf(filename, "%d %f %d %d %d\n", timestep, timestep*DELTATIME, population_size_old, A_counter, B_counter);
	}
}

void printTraitsPerIndividualToFile(FILE *filename){
	for(int i = 0; i < population_size_old; i++){
		fprintf(filename, "%d\t%f\t%f\n", individuals_old[i].phenotype, individuals_old[i].altruism, individuals_old[i].p);
	}
}

/**
 * Print to file the number of individuals, cumulative altruism level, and experienced altruism per cell.
 * Calls the printParametersToFile() function.
 */
void printPerCellStatistics(FILE *filename, int timestep){
	if(timestep == 0){
		printParametersToFile(filename);
		fprintf(filename, "Timestep Time Position X Y Density Cumulative_altruism_A Experienced_altruism\n");
	}
	if(timestep % OUTPUTINTERVAL == 0){
		for(int position = 0; position < NPOS; position++){
			fprintf(filename, "%d %f %d %f %f %f\n", timestep, timestep*DELTATIME, position, creal(density[position]), creal(altruism[position]), creal(normal_altruism_convolution[position]));
		}
	}
}

/**
 * Prints the experienced altruism per cell for a timestep in a tab-separated matrix that reflects the grid.
 * Should be called not more than once per timestep. Make sure to call createExperiencedAltruismMatrix() first.
 */
void printExperiencedAltruismMatrixToFile(){
	for(int index = 0; index < NPOS; index++){
		if(index != 0){
			if(index % N == 0){
				fprintf(expaltr_file, "\n");
			} else {
				fprintf(expaltr_file, "\t");
			}
		}
		fprintf(expaltr_file, "%f", creal(normal_altruism_convolution[index]));
	}
}

/**
 * Prints the density (number of individuals) per cell for a timestep in a tab-separated matrix that represents the grid.
 * Should be called not more than once per timestep. Make sure to call createDensityMatrix() first.
 */
void printDensityMatrixToFile(){
	for(int index = 0; index < NPOS; index++){
		if(index != 0){
			if(index % N == 0){
				fprintf(density_file, "\n");
				fprintf(density_file_A, "\n");
				fprintf(density_file_B, "\n");
			} else {
				fprintf(density_file, "\t");
				fprintf(density_file_A, "\t");
				fprintf(density_file_B, "\t");
			}
		}
		fprintf(density_file, "%d", (int)creal(density[index]));
		fprintf(density_file_A, "%d", density_A[index]);
		fprintf(density_file_B, "%d", density_B[index]);
	}
}

/**
 * Prints the summed altruism levels of the individuals per cell for a timestep in a tab-separated matrix that reflects the grid.
 * Should be called not more than once per timestep. Make sure to call createExperiencedAltruismMatrix() first, because this calls fillAltruismMatrix().
 */
void printSummedAltruismMatrixToFile(){
	for(int index = 0; index < NPOS; index++){
		if(index != 0){
			if(index % N == 0){
				fprintf(sumaltr_file, "\n");
				fprintf(sumaltr_file_A, "\n");
				fprintf(sumaltr_file_B, "\n");
			} else {
				fprintf(sumaltr_file, "\t");
				fprintf(sumaltr_file_A, "\t");
				fprintf(sumaltr_file_B, "\t");
			}
		}
		fprintf(sumaltr_file, "%f", sumaltr[index]);
		fprintf(sumaltr_file_A, "%f", creal(altruism[index])); //The summed altruism of As (phenotype 1) is in the altruism fftw object
		fprintf(sumaltr_file_B, "%f", sumaltr_B[index]);
	}
}

/**
 * Prints the summed p levels of the individuals per cell for a timestep in a tab-separated matrix that reflects the grid.
 * Should be called not more than once per timestep.
 */
void printSummedPmatrixToFile(){
	for(int index = 0; index < NPOS; index++){
		if(index != 0){
			if(index % N == 0){
				fprintf(sump_file, "\n");
				fprintf(sump_file_A, "\n");
				fprintf(sump_file_B, "\n");
			} else {
				fprintf(sump_file, "\t");
				fprintf(sump_file_A, "\t");
				fprintf(sump_file_B, "\t");
			}
		}
		fprintf(sump_file, "%f", sump[index]);
		fprintf(sump_file_A, "%f", sump_A[index]);
		fprintf(sump_file_B, "%f", sump_B[index]);
	}
}

/**
 * Creates a frequency table with the values of p and altruism level.
 */
void printTraitMatrixToFile(int bins_x, int bins_y, double binsize_x, double binsize_y){
	for(int i = 0; i < population_size_old; i++){
		int xbin = floor(individuals_old[i].altruism/binsize_x);
		if(xbin > (bins_x-1)){ //Add largest values to last bin
			xbin = bins_x-1;
		}
		int ybin = (bins_y - 1) - floor(individuals_old[i].p/binsize_y); //FIXME: This can give incorrect output if p is 0.2 or 0.3 (instead of 0.31) etc.
		if(ybin < 0){
			ybin = 0;
		}
		int position_in_matrix = ybin * bins_x + xbin;
		trait_matrix[position_in_matrix] += 1;
	}
	for(int index = 0; index < (bins_x*bins_y); index++){
		if(index != 0){
			if(index % bins_y == 0){
				fprintf(trait_matrix_file, "\n");
			}
			else{
				fprintf(trait_matrix_file, "\t");
			}
		}
		fprintf(trait_matrix_file, "%d", trait_matrix[index]);
	}
}

/**
 * Print the sum of the matrix before and after convolution with the normal kernel.
 */
void printSummedMatrixToFile(FILE *filename, int timestep){
	if(timestep == 0){
		printParametersToFile(filename);
		fprintf(filename, "Timestep Time sum_density sum_convolution_density sum_altruism sum_convolution_altruism\n");
	}
	if(timestep % OUTPUTINTERVAL == 0){
		fprintf(filename, "%d %f %f %f %f %f\n", timestep, timestep*DELTATIME, sumMatrix(density), sumMatrix(normal_density_convolution), sumMatrix(altruism), sumMatrix(normal_altruism_convolution));
	}
}

/**
 * Calculates the sum of the real parts of the elements of an FFTW complex object with NPOS elements. Useful to sum altruism or density matrix.
 */
double sumMatrix(fftw_complex* matrix){
	double sum = 0.0;
	for(int index = 0; index < NPOS; index++){
		if(cimag(matrix[index]) < -THRESHOLD || cimag(matrix[index]) > THRESHOLD){
			printf("\nWarning: sumMatrix() is used to sum real parts of fftw_complex object, but not all imaginary parts are 0.");
		}
		sum += creal(matrix[index]);
	}
	return sum;
}
