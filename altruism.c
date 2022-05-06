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
void MakeStepKernel(int); //This function is copied from Laura's code. Can perhaps be removed once createNormalKernel() is tested and working correctly
void createNormalKernel(int);
void makeIndividuals(void);
int rand(void);
void createLocalDensityMatrix(void);
void fillDensityMatrix(void);
void moveIndividual(int);
double calculateBirthRate(int);
double experiencedAltruism(int);
void reproduceIndividual(int);
void checkPopulationSize(void);
void updateStates(void);
void destroyFFTWplans(void);
void freeMemory(void);
double localDensity(int); //! Currently unused, can be removed once FFT is tested and working correctly !

//Define parameters TODO: Put in order of usage
//!! NB: Set INITIALB and INITIALP to 0 for initial model (without phenotypic differentiation) !!
#define TMAX 10
#define DELTATIME 1 //Q: Only used for output?
#define DELTASPACE 1
#define INITIALA 5
#define INITIALB 0
#define INITIALPOPULATIONSIZE INITIALA + INITIALB
#define XMAX 5
#define YMAX 5
#define NPOS XMAX * YMAX
#define INITIALP 1 //Set to 1 for only A offspring or 0 for only B offspring
#define MAXSIZE 1000 //Maximum number of individuals in the population. Note that MAXSIZE can be larger than XMAX*YMAX because multiple individuals are allowed at the same position.
#define DEATHRATE 0.1
#define BIRTHRATE 0.1 //Baseline max birth rate, birth rate for non-altruist
#define ALTRUISMSCALE 1 //Consider (2*SCALE + 1)^2 fields
#define COMPETITIONSCALE 1 //4
#define MOVEMENTSCALE 1
#define MOVE 0.5 //Probability to move in x direction = probability to move in y direction
#define B0 1 //Basal benefit of altruism
#define BMAX 5 //Maximum benefit of altruism
#define K 40 //Carrying capacity
#define THRESHOLD 0.000001 //Numbers lower than this are set to 0
#define FIELDS 4

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
fftw_complex* step_kernel;
fftw_complex* normal_kernel1D;
fftw_complex* normal_kernel2D;
fftw_complex* kernel_forward;
fftw_complex* density;
fftw_complex* density_forward;
fftw_complex* kernel_density_product;
fftw_complex* kernel_density_backward;

fftw_plan fftw_plan_kernel;
fftw_plan fftw_plan_density;
fftw_plan fftw_plan_kernel_density;

struct Individual* individuals_old; //This is the 'old state'
struct Individual* individuals_new; //This is the 'new state'

//Main
int main() {
	srand(time(0));
	init_genrand64(time(0));
	allocateMemory();
	createFFTWplans();
	createNormalKernel(COMPETITIONSCALE); //Create and execute only once, same for each timestep
	fftw_execute(fftw_plan_kernel);
	makeIndividuals();
	population_size_old = INITIALPOPULATIONSIZE;
	population_size_new = 0;
    for (int t = 0; t < TMAX; t++) {
    	createLocalDensityMatrix();
		for (int i = 0; i < population_size_old; i++){
			double probabilityOfEvent = genrand64_real2();
			if (probabilityOfEvent > DEATHRATE){ //If individual does NOT die...
				moveIndividual(i); //...Move it
				individuals_new[i] = individuals_old[i];
				population_size_new += 1; //...And add it to the population size of the new state.
				double birth_rate = calculateBirthRate(i);
				if (probabilityOfEvent < DEATHRATE + birth_rate){
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
    individuals_old = malloc(MAXSIZE * sizeof(*individuals_old));
    individuals_new = malloc(MAXSIZE * sizeof(*individuals_new));
    if (individuals_old == NULL) {
        printf("ERROR: Memory for individuals_old not allocated.\n");
        exit(1);
    }
    else if (individuals_new == NULL) {
        printf("ERROR: Memory for individuals_new not allocated.\n");
        exit(1);
    }
    step_kernel = fftw_alloc_complex(NPOS);
	normal_kernel1D = fftw_alloc_complex(XMAX);
	normal_kernel2D = fftw_alloc_complex(NPOS);
    kernel_forward = fftw_alloc_complex(NPOS);
    density = fftw_alloc_complex(NPOS);
    density_forward = fftw_alloc_complex(NPOS);
    kernel_density_product = fftw_alloc_complex(NPOS);
    kernel_density_backward = fftw_alloc_complex(NPOS);
}

/**
 * Creates plans for the Fourier transformations used in the code.
 */
void createFFTWplans(void){
	fftw_plan_kernel = fftw_plan_dft_2d(XMAX, YMAX, normal_kernel2D, kernel_forward, FFTW_FORWARD, FFTW_MEASURE);
	fftw_plan_density = fftw_plan_dft_2d(XMAX, YMAX, density, density_forward, FFTW_FORWARD, FFTW_MEASURE);
	fftw_plan_kernel_density = fftw_plan_dft_2d(XMAX, YMAX, kernel_density_product, kernel_density_backward, FFTW_BACKWARD, FFTW_MEASURE);
}

//*** Code to make kernel function, taken (and slightly modified) from Laura's code. Can possibly be removed once normal function is tested and working correctly ***//

/* This function makes the kernel using a step function. Code modified from Hilje's code.
 * If the number of rows and columns are not the same, the kernel function will form an ellipse instead of a circle. */
void MakeStepKernel(int scale)
{
    int x,y,dx,dy; // x and y will denote the 'x-coordinate' and 'y-coordinate' of the matrix, respectively, and dx, dy the distance to the focal point in the x- and y-direction, respectively.
    double totsum = 0.0; // counter for summation of kernel values, will use this for normalisation later

    // Fill the kernel, unnormalised
    for(int l = 0; l < NPOS; l++) //Note: Always loop over same number as used in memory allocation!
    {
        // This looping over indices gives row-major order.
        y = l/XMAX;     // int division! Always rounding down: this gives the quotient, which is the 'y coordinate'
        x = l - y*XMAX; // this is the rest/leftover or 'x coordinate'
        dx = (int) fmin(x, XMAX - x); // distance to focal point in x direction. Casting to int because we're only working with ints here, so the result should be an int anyway.
        dy = (int) fmin(y, YMAX - y); // distance to focal point in y direction. Casting to int because we're only working with ints here, so the result should be an int anyway.
        if( dx*dx + dy*dy < ( (int) (scale*DELTASPACE) * (scale*DELTASPACE) ) + 1 ) // Pythagoras
        {
            step_kernel[l] = 1.0 + 0.0 * I;
            totsum = totsum + 1.0;
        }
        else
            step_kernel[l] = 0.0 + 0.0 * I;
    }

    // Normalise the kernel
    for(int l = 0; l < NPOS; l++)
    {
        step_kernel[l] = (DELTASPACE*DELTASPACE)*step_kernel[l] / totsum;
    }
}

//*** End of part taken from Laura's code ***//

/**
 * Creates a normal kernel. Numbers lower than THRESHOLD are set to 0. This should equal line 34-61 in the Fortran kernels.f90 file.
 */
void createNormalKernel(int scale){
	double preFactor = 1.0/(2.0*(scale*DELTASPACE)*(scale*DELTASPACE));
	for(int x = 0; x < XMAX; x++){
		double exp_counter = 0.0;
		for(int field = -FIELDS; field < FIELDS; field++){
			exp_counter += exp(-preFactor * ((x + field*XMAX)*(x + field*XMAX)));
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
		normal_kernel2D[i] = DELTASPACE*DELTASPACE * normal_kernel2D[i]/kernel_sum;
	}
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
 * Creates a matrix with the local density at each position in the field using convolution of the kernel created above and the density matrix created in fillDensityMatrix().
 */
void createLocalDensityMatrix(){
	fillDensityMatrix();
	fftw_execute(fftw_plan_density); //Forward Fourier of density matrix
	for (int position = 0; position < NPOS; position++){ //Multiply forward-transformed kernel and density matrices
	    kernel_density_product[position] = kernel_forward[position] * density_forward[position];
	}
	fftw_execute(fftw_plan_kernel_density); //Convolution: Inverse Fourier of product of forward-transformed kernel and density matrices
	for (int element; element < NPOS; element++){
		kernel_density_backward[element] = kernel_density_backward[element]/NPOS; //Output of FFTW_BACKWARD is automatically multiplied by number of elements, so divide by number of elements (NPOS) to get local density
	}
}

/**
 * Creates a matrix with the absolute number of individuals at each position in the field.
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
 * Assigns a new position in the field to the input individual.
 * i: The individual to move.
 */
void moveIndividual(int i){ //TODO: Try using modulo here
	double move_x = genrand64_real2();
	double move_y = genrand64_real2();
	if (move_x < MOVE){
		double random = genrand64_real2();
		if (random < 0.5){
			if (individuals_old[i].xpos+MOVEMENTSCALE*DELTASPACE <= XMAX){
				individuals_new[i].xpos = individuals_old[i].xpos+MOVEMENTSCALE*DELTASPACE;
			}
		}
		else if (individuals_old[i].xpos-MOVEMENTSCALE*DELTASPACE > 0){
			individuals_new[i].xpos = individuals_old[i].xpos-MOVEMENTSCALE*DELTASPACE;
		}
	}
	if (move_y < MOVE){
		double random = genrand64_real2();
		if (random < 0.5){
			if (individuals_old[i].ypos+MOVEMENTSCALE*DELTASPACE <= YMAX){
				individuals_new[i].ypos = individuals_old[i].ypos+MOVEMENTSCALE*DELTASPACE;
			}
		}
		else if (individuals_old[i].ypos-MOVEMENTSCALE*DELTASPACE > 0){
			individuals_new[i].ypos = individuals_old[i].ypos-MOVEMENTSCALE*DELTASPACE;
		}
	}
}

/**
 * Calculates the birth rate of the input individual.
 * i: The individual whose birth rate is calculated.
 * returns: The birth rate of the individual.
 */
double calculateBirthRate(int i){
	double experienced_altruism = experiencedAltruism(i);
		int position = individuals_old[i].xpos * XMAX + individuals_old[i].ypos; //Convert x and y coordinates of individual to find corresponding position in fftw_complex object
		double local_density = kernel_density_backward[position];
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
 * Calculates the level of altruism experienced by the input individual. TODO: Use fftw here to calculate convolution of positions + kernel to derive experienced altruism
 * i: The individual whose experienced altruism is calculated.
 * returns: The level of altruism experienced by the input individual.
 */
double experiencedAltruism(int i){
	int x_individual = individuals_old[i].xpos;
	int y_individual = individuals_old[i].ypos;
	double cumulative_altruism = 0.0;
	//Note that multiple individuals can be present in one x,y field. Non-existing fields are now also checked. But doesn't matter for now because it should be done using FFT anyway.
	for (int x = x_individual-ALTRUISMSCALE*DELTASPACE; x < x_individual+ALTRUISMSCALE*DELTASPACE; x++){ //Loop over all x,y positions within scale from individual
		for (int y = y_individual-ALTRUISMSCALE*DELTASPACE; y < y_individual+ALTRUISMSCALE*DELTASPACE; y++){
			for (int potential_neighbor_index = 0; potential_neighbor_index < MAXSIZE; potential_neighbor_index++){ //Loop over all individuals
				struct Individual potential_neighbor = individuals_old[potential_neighbor_index];
				if (potential_neighbor.phenotype == 0){ //Only neighbors with phenotype 0 (A) can contribute
					if (potential_neighbor.xpos == x && potential_neighbor.ypos == y){
						cumulative_altruism += potential_neighbor.altruism; //If so, get their level of altruism
					}
				}
			}
		}
	}
	return cumulative_altruism;
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
	memset(individuals_old, 0, MAXSIZE * sizeof(*individuals_old)); //This resets the individuals_old array, setting all struct variables to 0
	for(int i = 0; i < population_size_new; i++){ //New individuals become old individuals TODO: Can this be made more efficient? Do something smart with pointers?
		individuals_old[i] = individuals_new[i];
	}
	memset(individuals_new, 0, MAXSIZE * sizeof(*individuals_new)); //Reset the individuals_new array
	population_size_old = population_size_new; //Switch to new state, including all individuals that were born or didn't die in the previous timestep
	population_size_new = 0; //Reset value of new state population size
	memset(density, 0, NPOS * sizeof(*density));
	memset(density_forward, 0, NPOS * sizeof(*density_forward));
	memset(kernel_density_product, 0, NPOS * sizeof(*kernel_density_product));
	memset(kernel_density_backward, 0, NPOS * sizeof(*kernel_density_backward));
}

/**
 * Destroys all plans created for the Fourier transformations in createFFTWplans().
 */
void destroyFFTWplans(void){
	fftw_destroy_plan(fftw_plan_kernel);
	fftw_destroy_plan(fftw_plan_density);
	fftw_destroy_plan(fftw_plan_kernel_density);
}

/**
 * Frees all memory allocated for the arrays and fftw_complex objects in allocateMemory().
 */
void freeMemory(void){
	free(individuals_old);
	free(individuals_new);
	fftw_free(step_kernel);
	fftw_free(kernel_forward);
	fftw_free(density);
	fftw_free(density_forward);
	fftw_free(kernel_density_product);
	fftw_free(kernel_density_backward);
}

/**
 * Old function to calculate local density experienced by input individual, roughly using a step function. To be removed once Fourier transform is tested and working correctly.
 * i: The input individual.
 */
double localDensity(int i){
	int x_individual = individuals_old[i].xpos;
	int y_individual = individuals_old[i].ypos;
	double cumulative_density = 0.0;
	for (int x = x_individual-COMPETITIONSCALE*DELTASPACE; x < x_individual+COMPETITIONSCALE*DELTASPACE; x++){
		for (int y = y_individual-COMPETITIONSCALE*DELTASPACE; y < y_individual+COMPETITIONSCALE*DELTASPACE; y++){
			for (int potential_neighbor_index = 0; potential_neighbor_index < MAXSIZE; potential_neighbor_index++){
				struct Individual potential_neighbor = individuals_old[potential_neighbor_index];
				if (potential_neighbor.xpos == x && potential_neighbor.ypos == y){
						cumulative_density += 1; //Should perhaps be scaled
				}
			}
		}
	}
	return cumulative_density;
}

