/*
 * altruism.c
 *
 *  Created on: 2 mrt. 2022
 *      Author: Irene Bouwman
 */

//Include
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include "mt64.h"
#include <complex.h>
#include <fftw3.h>


//Declare functions (in order of usage)
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
void checkPopulationSize(void);
void updateStates(void);
void destroyFFTWplans(void);
void freeMemory(void);

//Define parameters TODO: Put in order of usage
//!! NB: Set INITIALB  to 0 and INITIALP to 1 for initial model (without phenotypic differentiation) !!
#define TMAX 10
#define DELTATIME 0.1 //Multiply rate by DELTATIME to get probability per timestep
#define DELTASPACE 1 //Size of a position. This equals 1/resolution in the Fortran code.
#define INITIALA 5
#define INITIALB 0
#define INITIALPOPULATIONSIZE INITIALA + INITIALB
#define XMAX 5
#define YMAX 5
#define NPOS XMAX * YMAX
#define INITIALP 1 //Set to 1 for only A offspring or 0 for only B offspring
#define MAXSIZE 1000 //Maximum number of individuals in the population. Note that MAXSIZE can be larger than XMAX*YMAX because multiple individuals are allowed at the same position.
#define DEATHRATE 1
#define BIRTHRATE 1 //Baseline max birth rate, birth rate for non-altruist
#define ALTRUISMSCALE 1 //Consider (2*SCALE + 1)^2 fields
#define COMPETITIONSCALE 1 //4
#define MOVEMENTSCALE 1
#define MOVE 0.5 //Probability to move in x direction = probability to move in y direction
#define B0 1 //Basal benefit of altruism
#define BMAX 5 //Maximum benefit of altruism
#define K 40 //Carrying capacity
#define THRESHOLD 0.0000000001 //Numbers lower than this are set to 0
#define FIELDS 7 //Number of fields to take into account (in each direction) when creating the normal kernel

//Declare structures
struct Individual {
	int xpos;
	int ypos;
	double altruism;
	double p;
	int phenotype; //0 is A, 1 is B
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


//Main
int main() {
	srand(time(0));
	init_genrand64(time(0));
	allocateMemory();
	createFFTWplans();
	createNormalKernel(COMPETITIONSCALE, normal_for_density); //Create and execute only once, same for each timestep
	createNormalKernel(ALTRUISMSCALE, normal_for_altruism);
	fftw_execute(fftw_plan_normal_for_density_forward);
	fftw_execute(fftw_plan_normal_for_altruism_forward);
	makeIndividuals();
	population_size_old = INITIALPOPULATIONSIZE;
	population_size_new = 0;
    for (int t = 0; t < TMAX; t++) {
    	createLocalDensityMatrix();
    	createExperiencedAltruismMatrix();
		for (int i = 0; i < population_size_old; i++){
			double probabilityOfEvent = genrand64_real2();
			if (probabilityOfEvent > DEATHRATE*DELTATIME){ //If individual does NOT die...
				moveIndividual(i); //...Move it
				individuals_new[i] = individuals_old[i];
				population_size_new += 1; //...And add it to the population size of the new state.
				double birth_rate = calculateBirthRate(i);
				if (probabilityOfEvent < DEATHRATE*DELTATIME + birth_rate*DELTATIME){
					reproduceIndividual(i);
					moveIndividual(i+1);
				}
			}
		}
    	checkPopulationSize();
    	updateStates(); //New state becomes old state
   }
   destroyFFTWplans();
   freeMemory();
   return 0;
}

/**
 * Allocates memory for the arrays and fftw_complex objects used in the code.
 */
void allocateMemory(void){
    individuals_old = malloc(MAXSIZE * sizeof(struct Individual));
    individuals_new = malloc(MAXSIZE * sizeof(struct Individual));
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
	for (int i = 0; i < INITIALPOPULATIONSIZE; i++){ //First fill in all initial parameters that are the same for As and Bs
		individuals_old[i].xpos = rand() % XMAX+1;;
		individuals_old[i].ypos = rand() % YMAX+1;
		individuals_old[i].altruism = genrand64_real2(); //TODO: Once evolution (mutation) is implemented, this should probably be 0. Now random to at least have variation.
		individuals_old[i].p = INITIALP;
	}
	for (int i = 0; i < INITIALA; i++){
		individuals_old[i].phenotype = 0;
	}
	for (int i = INITIALA; i < INITIALPOPULATIONSIZE; i++){ //Put Bs after As so nothing is overwritten (Bs = Population size - As)
		individuals_old[i].phenotype = 1;
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
	for (int x = 1; x < XMAX+1; x++){ //Add 1 to let x and y correspond to actual x and y positions of the individuals
		for (int y = 1; y < YMAX+1; y++){
			int counter = 0;
			for (int index = 0; index < population_size_old; index++){
				if (individuals_old[index].xpos == x & individuals_old[index].ypos == y){
					counter += 1;
				}
			}
			density[position] = creal(counter);
			position += 1;
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
 * Assigns a new position in the field to the input individual. TODO: Change/replace this function to use a diffusion process (using DELTATIME)
 * i: The individual to move.
 */
void moveIndividual(int i){ //TODO: Try using modulo here
	double move_x = genrand64_real2();
	double move_y = genrand64_real2();
	if (move_x < MOVE){
		double random = genrand64_real2();
		if (random < 0.5){
			if (individuals_old[i].xpos+MOVEMENTSCALE*(1/DELTASPACE) <= XMAX){
				individuals_new[i].xpos = individuals_old[i].xpos+MOVEMENTSCALE*(1/DELTASPACE);
			}
		}
		else if (individuals_old[i].xpos-MOVEMENTSCALE*(1/DELTASPACE) > 0){
			individuals_new[i].xpos = individuals_old[i].xpos-MOVEMENTSCALE*(1/DELTASPACE);
		}
	}
	if (move_y < MOVE){
		double random = genrand64_real2();
		if (random < 0.5){
			if (individuals_old[i].ypos+MOVEMENTSCALE*(1/DELTASPACE) <= YMAX){
				individuals_new[i].ypos = individuals_old[i].ypos+MOVEMENTSCALE*(1/DELTASPACE);
			}
		}
		else if (individuals_old[i].ypos-MOVEMENTSCALE*(1/DELTASPACE) > 0){
			individuals_new[i].ypos = individuals_old[i].ypos-MOVEMENTSCALE*(1/DELTASPACE);
		}
	}
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
	double birth_rate;
	if (individuals_old[i].phenotype == 0){ //Only individuals with phenotype 0 (A) pay altruism cost
		birth_rate = BIRTHRATE * (1.0 - individuals_old[i].altruism + benefit) * (1.0 - (local_density/K));
	}
	else{
		birth_rate = BIRTHRATE * (1.0 + benefit) * (1.0 - (local_density/K));
	}
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
	individuals_new[i+1] = individuals_old[i]; //Initially child = parent BUT overwrite phenotype below
	population_size_new += 1;
	double random_phenotype = genrand64_real2();
	if (random_phenotype < individuals_old[i].p){ //p is probability that child has phenotype A (0)
		individuals_new[i+1].phenotype = 0;
	}
	else{
		individuals_new[i+1].phenotype = 1;
	}
}

/**
 * Checks whether the population size is not out of bounds.
 * Stops run and throws error when population size is above MAXSIZE or below 0.
 * Stops run and prints message when population size is 0 i.e. population died out.
 */
void checkPopulationSize(){
	if ((population_size_new > MAXSIZE) || population_size_new < 0){
		printf("\nERROR: Population size must be between 0 and %d, but population size for next timestep is %d.\n", MAXSIZE, population_size_new);
		exit(1);
	}
	else if (population_size_new == 0){
		printf("\n%d individuals left in next timestep. Population died out!\n", population_size_new);
		exit(1);
	}
}

/**
 * Updates the old and new state for the next timestep, i.e. old individuals are replaced by new individuals. Resets all pointers with timestep-specific information.
 */
void updateStates(){
	individuals_old_ptr = &individuals_old; //Make pointers to pointers, this is necessary to swap old and new individuals
	individuals_new_ptr = &individuals_new;
	memset(individuals_old, 0, MAXSIZE * sizeof(*individuals_old)); //Delete the old individuals, i.e. set array to 0
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
