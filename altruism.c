/*
 * altruism.c
 *
 *  Created on: 2 mrt. 2022
 *      Author: Irene Bouwman
 */

//Include
# include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include "mt64.h"
#include <complex.h>
#include <fftw3.h>
#include "ziggurat.h"
#include "random.h"

//Declare functions (in order of usage)
//Functions used in main():
void srand(unsigned int seed);
void allocateMemory(void);
void createFFTWplans(void);
void createNormalKernel(int, fftw_complex*);
void makeIndividuals(void);
int rand(void);
void createLocalDensityMatrix(void);
void fillDensityMatrix(void);
void createExperiencedAltruismMatrix(void);
void fillAltruismMatrix(void);
void moveIndividual(int);
double calculateBirthRate(int);
void reproduceIndividual(int);
double randomExponential(void);
void checkPopulationSize(int);
void updateStates(void);
void destroyFFTWplans(void);
void freeMemory(void);
//Functions used to create output files:
void printParametersToFile(FILE*);
void printMeanAltruismToFile(FILE*, int);
void printPopulationSizeToFile(FILE*, int);
void printPerCellStatistics(FILE*, int);
void printSummedMatrixToFile(FILE*, int);
double sumMatrix(fftw_complex*);

//Define parameters and settings, following 2D/parameters in the Fortran code (and Table 1 of the paper)
//Settings
#define TMAX 101
#define OUTPUTINTERVAL 50 //Number of timesteps between each output print
#define FIELDS 7 //Number of fields to take into account (in each direction) when creating the normal kernel
#define DELTATIME 0.08 //Multiply rate by DELTATIME to get probability per timestep
#define DELTASPACE 1.0 //Size of a position. This equals 1/resolution in the Fortran code.
#define STEADYSTATEDENSITY (1 - DEATHRATE/BIRTHRATE) * K
#define GRIDSIZE (XMAX * DELTASPACE) * (YMAX * DELTASPACE)
#define INITIALALTRUISM 0.0
#define THRESHOLD 0.0000000001 //Numbers lower than this are set to 0
//Parameters
#define BIRTHRATE 5.0 //Baseline max birth rate
#define DEATHRATE 1.0
#define MUTATIONPROBABILITY 0.001
#define MEANMUTSIZEALTRUISM 0.005
#define ALTRUISMSCALE 1
#define COMPETITIONSCALE 4
#define DIFFUSIONCONSTANT 0.04
#define MOVEMENTSCALE sqrt(2 * DIFFUSIONCONSTANT * DELTATIME)
#define K 40 //Carrying capacity
#define B0 1.0 //Basal benefit of altruism
#define BMAX 5.0 //Maximum benefit of altruism
#define XMAX 512
#define YMAX XMAX //The arena must be a square; XMAX and YMAX are used for code readability
#define NPOS XMAX * YMAX

//Declare structures
struct Individual {
	int xpos;
	int ypos;
	double altruism;
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

int INITIALPOPULATIONSIZE = round(STEADYSTATEDENSITY * GRIDSIZE); //Initial and maximal population size depend on steady state density and grid size.
int MAXPOPULATIONSIZE = round(1.5 * K * GRIDSIZE); //Note that MAXPOPULATIONSIZE can be larger than NPOS because multiple individuals are allowed at the same position.
int newborns;
int deaths;
int i_new;

//Main
int main() {
	time_t tm;
	time(&tm);
	printf("Running %s main branch. Started at %s\n", __FILE__, ctime(&tm));
	srand(time(0));
	init_genrand64(time(0));
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
	FILE *outputfile;
	outputfile = fopen("filename.txt", "w+");
    for (int t = 0; t < TMAX; t++) {
    	if(t == 0){
    		printf("Simulation has started!\nProgress (printed every %d timesteps):\n", OUTPUTINTERVAL);
    	}
    	if(t % OUTPUTINTERVAL == 0){
    		printf("%d out of %d timesteps.\n", t, TMAX);
    	}
    	//printMeanAltruismToFile(outputfile, t);
    	//printPopulationSizeToFile(outputfile, t);
    	newborns = 0;
		deaths = 0;
    	createLocalDensityMatrix();
    	createExperiencedAltruismMatrix();
    	printSummedMatrixToFile(outputfile, t);
    	//printPerCellStatistics(outputfile, t);
		for (int i = 0; i < population_size_old; i++){
			i_new = i + newborns - deaths; //The index of i in the new timestep, taking into account births and deaths the current timestep
			double probabilityOfEvent = genrand64_real2();
			if (probabilityOfEvent < DEATHRATE*DELTATIME){ //If individual dies...
				deaths += 1;
			}
			else{ //If individual doesn't die...
				//moveIndividual(i); //...Move it
				individuals_new[i_new] = individuals_old[i];
				population_size_new += 1; //...And add it to the population size of the new state.
				double birth_rate = calculateBirthRate(i);
				if (probabilityOfEvent < DEATHRATE*DELTATIME + birth_rate*DELTATIME){ //If the individual reproduces...
					reproduceIndividual(i); //...Create the child
					population_size_new += 1; //...And add it to the population size of the next timestep
					//moveIndividual(i_new + 1); //Move the child: Index of child = index of parent in the new state + 1
					newborns += 1;
				}
			}
		}
    	checkPopulationSize(t);
    	updateStates(); //New state becomes old state
   }
   destroyFFTWplans();
   freeMemory();
   printf("\nDone.\n");
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
	fftw_plan_normal_for_density_forward = fftw_plan_dft_2d(XMAX, YMAX, normal_for_density, normal_for_density_forward, FFTW_FORWARD, FFTW_MEASURE);
	fftw_plan_normal_for_altruism_forward = fftw_plan_dft_2d(XMAX, YMAX, normal_for_altruism, normal_for_altruism_forward, FFTW_FORWARD, FFTW_MEASURE);
	fftw_plan_density_forward = fftw_plan_dft_2d(XMAX, YMAX, density, density_forward, FFTW_FORWARD, FFTW_MEASURE);
	fftw_plan_normal_density_convolution = fftw_plan_dft_2d(XMAX, YMAX, normal_forward_density_forward_product, normal_density_convolution, FFTW_BACKWARD, FFTW_MEASURE);
	fftw_plan_altruism_forward = fftw_plan_dft_2d(XMAX, YMAX, altruism, altruism_forward, FFTW_FORWARD, FFTW_MEASURE);
	fftw_plan_normal_altruism_convolution = fftw_plan_dft_2d(XMAX, YMAX, normal_forward_altruism_forward_product, normal_altruism_convolution, FFTW_BACKWARD, FFTW_MEASURE);
}


/**
 * Creates a normal kernel. Numbers lower than THRESHOLD are set to 0. This should equal line 34-61 in the Fortran kernels.f90 file.
 */
void createNormalKernel(int scale, fftw_complex* normal_kernel2D){
	fftw_complex* normal_kernel1D;
	normal_kernel1D = fftw_alloc_complex(XMAX);
	double preFactor = 1.0/(2.0*(scale*(1/DELTASPACE))*(scale*(1/DELTASPACE)));
	for(int x = 0; x < XMAX; x++){
		double exp_counter = 0.0;
		for(int field = -FIELDS; field < FIELDS; field++){
			exp_counter += exp(-preFactor * ((x + field*XMAX)*(x + field*XMAX))); //TODO: Change summation strategy so small numbers don't disappear
		}
		if(exp_counter < THRESHOLD){
			exp_counter = 0;
		}
		normal_kernel1D[x] = creal(exp_counter); //Fortran uses partial summation, which is slightly different
	}
	int index = 0;
	double kernel_sum = 0.0;
	for(int x = 0; x < XMAX; x++){
		for(int y = 0; y < YMAX; y++){
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
		individuals_old[i].xpos = rand() % XMAX+1;;
		individuals_old[i].ypos = rand() % YMAX+1;
		individuals_old[i].altruism = INITIALALTRUISM;
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
	int position = 0;
	int sum_counter = 0;
	for (int x = 1; x < XMAX+1; x++){ //Add 1 to let x and y correspond to actual x and y positions of the individuals
		for (int y = 1; y < YMAX+1; y++){
			int counter = 0;
			for (int index = 0; index < population_size_old; index++){
				if (individuals_old[index].xpos == x & individuals_old[index].ypos == y){
					counter += 1;
					sum_counter += 1;
				}
			}
			density[position] = creal(counter);
			position += 1;
		}
	}
	if(sum_counter != population_size_old){
		printf("Problem: number of individuals in density matrix (%d) doesn't match population size (%d)!\n", sum_counter, population_size_old); //TODO: This can be removed after all required tests are successful
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
 * Creates a matrix with for each position in the field the cumulative level of altruism of the individuals at that position.
 * Similar to the creation of the density matrix.
 */
void fillAltruismMatrix(){
	int position = 0;
	for (int x = 1; x < XMAX+1; x++){
		for (int y = 1; y < YMAX+1; y++){
			double cumulative_altruism = 0.0;
			for (int index = 0; index < population_size_old; index++){
				if (individuals_old[index].xpos == x & individuals_old[index].ypos == y){
					cumulative_altruism += individuals_old[index].altruism;
				}
			}
			altruism[position] = creal(cumulative_altruism);
			position += 1;
		}
	}
}

/**
 * Assigns a new position in the field to the input individual.
 * i: The individual to move.
 */
void moveIndividual(int i){ //TODO: Try using modulo here
	void * r = random_new(time(NULL));
	int move_x = round(random_normal(r, 0, MOVEMENTSCALE/DELTASPACE));
	int move_y = round(random_normal(r, 0, MOVEMENTSCALE/DELTASPACE));
	individuals_new[i].xpos = ((individuals_old[i].xpos + move_x + XMAX -1) % XMAX)+1;
	individuals_new[i].ypos = ((individuals_old[i].ypos + move_y + YMAX -1) % YMAX)+1;
}

/**
 * Calculates the birth rate of the input individual.
 * i: The individual whose birth rate is calculated.
 * returns: The birth rate of the individual.
 */
double calculateBirthRate(int i){
	int position = individuals_old[i].xpos * XMAX + individuals_old[i].ypos; //Convert x and y coordinates of individual to find corresponding position in fftw_complex object
	double local_density = normal_density_convolution[position];
	double experienced_altruism = normal_altruism_convolution[position];
	double benefit = (BMAX * experienced_altruism)/((BMAX/B0) + experienced_altruism);
	double birth_rate = BIRTHRATE * (1.0 - individuals_old[i].altruism + benefit) * (1.0 - (local_density/K));
	if (birth_rate < 0){
		birth_rate = 0; //Negative birth rates are set to 0
	}
	return birth_rate;
}

/**
 * Creates a child of individual i at index i+1 in the individuals_new array.
 * i: The parent individual.
 */
void reproduceIndividual(int i){
	individuals_new[i_new+1] = individuals_old[i]; //Initially child = parent BUT consider mutation below
	double random_altruism = genrand64_real2();
	if(random_altruism < MUTATIONPROBABILITY){ //If mutation occurs...
		double delta_altruism;
		double random_direction = genrand64_real2(); //...Randomly decide direction of mutation
		if(random_direction < 0.5){
			delta_altruism = -MEANMUTSIZEALTRUISM * randomExponential(); //...Calculate change in altruism level due to mutation
		}
		else{
			delta_altruism = MEANMUTSIZEALTRUISM * randomExponential();
		}
		individuals_new[i_new+1].altruism = individuals_old[i].altruism + delta_altruism; //...Calculate altruism level of child
		if(individuals_new[i_new+1].altruism < 0){
			individuals_new[i_new+1].altruism = 0.0;
		}
	}
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
 * Prints predefined parameters to the input file.
 */
void printParametersToFile(FILE *filename){
	time_t tm;
	time(&tm);
	fprintf(filename, "Simulation from %s performed at %s", __FILE__, ctime(&tm));
	fprintf(filename, "Predefined parameters:\n");
	fprintf(filename, "Tmax = %d\n", TMAX);
	fprintf(filename, "DeltaTime = %f\n", DELTATIME);
	fprintf(filename, "DeltaSpace = %f\n", DELTASPACE);
	fprintf(filename, "Initial population size = %d\n", INITIALPOPULATIONSIZE);
	fprintf(filename, "Number of positions = %d\n", NPOS);
	fprintf(filename, "Initial altruism level = %f\n", INITIALALTRUISM);
	fprintf(filename, "Death rate = %f\n", DEATHRATE);
	fprintf(filename, "Birth rate = %f\n", BIRTHRATE);
	fprintf(filename, "Mutation probability altruism = %f\n", MUTATIONPROBABILITY);
	fprintf(filename, "Mean mutation size altruism = %f\n", MEANMUTSIZEALTRUISM);
	fprintf(filename, "Scale of altruism = %d\n", ALTRUISMSCALE);
	fprintf(filename, "Scale of competition = %d\n", COMPETITIONSCALE);
	fprintf(filename, "Scale of movement = %f\n", MOVEMENTSCALE);
	fprintf(filename, "B0 = %f\n", B0);
	fprintf(filename, "BMAX = %f\n", BMAX);
	fprintf(filename, "K = %d\n", K);
	fprintf(filename, "Number of fields (kernel) = %d\n", FIELDS);
}

/**
 * Calculates the mean level of altruism in the population and prints this to an outputfile.
 * Calls the printParametersToFile() function.
 */
void printMeanAltruismToFile(FILE *filename, int timestep){
	if(timestep == 0){
		printParametersToFile(filename);
		fprintf(filename, "Timestep Time Mean_altruism_level\n");
	}
	if(timestep % OUTPUTINTERVAL == 0){
		double cumulative_altruism;
		for(int i = 0; i < population_size_old; i++){
			cumulative_altruism += individuals_old[i].altruism;
		}
		double mean_altruism = cumulative_altruism/population_size_old;
		fprintf(filename, "%d %f %f\n", timestep, timestep*DELTATIME, mean_altruism);
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
 * Print to file the number of individuals, cumulative altruism level, and experienced altruism per cell.
 * Calls the printParametersToFile() function.
 */
void printPerCellStatistics(FILE *filename, int timestep){
	if(timestep == 0){
		printParametersToFile(filename);
		fprintf(filename, "Timestep Time Position X Y Density Cumulative_altruism Experienced_altruism\n");
	}
	if(timestep % OUTPUTINTERVAL == 0){
		for(int position = 0; position < NPOS; position++){
			int x = floor(position/XMAX) + 1; //TODO: Maybe make functions to convert from x and y to position and back
			int y = (position % XMAX) + 1;
			fprintf(filename, "%d %f %d %d %d %f %f %f\n", timestep, timestep*DELTATIME, position, x, y, creal(density[position]), creal(altruism[position]), creal(normal_altruism_convolution[position]));
		}
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
