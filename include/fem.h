#include <strings.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

#include "Accelerate/Accelerate.h"

#include "Mesh.h"

/* prototypes for storage functions (Utils.c) */
double **BuildMatrix(int Nrows, int Ncols);
double  *BuildVector(int Nrows);
int    **BuildIntMatrix(int Nrows, int Ncols);
int     *BuildIntVector(int Nrows);

double **DestroyMatrix(double **);
double  *DestroyVector(double *);
int    **DestroyIntMatrix(int **);
int     *DestroyIntVector(int *);

void PrintMatrix(char *message, double **A, int Nrows, int Ncols);
void SaveMatrix(char *filename, double **A, int Nrows, int Ncols);

/* geometric/mesh functions */
Mesh *ReadMesh3d(char *filename);

void  PrintMesh ( Mesh *mesh );

void Normals3d(Mesh *mesh, int k, double *nx, double *ny, double *nz, double *sJ);

void GeometricFactors3d(Mesh *mesh, int k,
		      double *drdx, double *dsdx, double *dtdx, 
		      double *drdy, double *dsdy, double *dtdy, 
		      double *drdz, double *dsdz, double *dtdz, 
			double *J);

/* start up */
void StartUp3d(Mesh *mesh);

void BuildMaps3d(Mesh *mesh);

/* Parallel */
void LoadBalance3d(Mesh *mesh);

void FacePair3d(Mesh *mesh, int *maxNv);

void ParallelPairs(void *objs, int Nmyobjs, int sizeobj,
		   int  (*numget)(const void *),
		   void (*numset)(const void *, int ),
		   int  (*procget)(const void *),
		   void (*marry)(const void *, const void *),
		   int (*compare_objs)(const void *, const void *));

void MaxwellsMPISend3d(Mesh *mesh, float *Q);
void MaxwellsMPIRecv3d(Mesh *mesh, float *partQ);
void MaxwellsMPISend3d_restruct(Mesh *mesh, float *Q2, int *Q_to_Q2);

/* Maxwells functions */
void MaxwellsRun3d(Mesh *mesh, double FinalTime, double dt, double *Hx, double *Hy, double *Hz, double *Ex, double *Ey, double *Ez, float *vgeo, float *mapinfo, float *surfinfo);

void volumeKernel(Mesh *mesh, float *Q, float *rhsQ, float *vgeo);

void surfaceKernel(Mesh *mesh, float *Q, float *partQ, float *rhsQ, float *mapinfo, float *surfinfo);

void rkKernel(Mesh *mesh, float *resQ, float *rhsQ, float *Q, float frka, float frkb, float fdt);

/* Maxwells functions with BLAS */
void optimised_volumeKernel(Mesh *mesh, float *Q, float *rhsQ, float *vgeo, float *Dr, float *Ds, float *Dt, float *kField, float *DRF, float *DSF, float *DTF);

void optimised_surfaceKernel(Mesh *mesh, float *Q, float *partQ, float *rhsQ, float *mapinfo, float *surfinfo, float *LIFT, float *kFlux, float *kField);

/* Maxwells functions with BLAS and restructured data */
void optimised_volumeKernel_restruct(Mesh *mesh, float *Q2, float *rhsQ2, float *vgeo, float *Dr, float *Ds, float *Dt, float *DRF, float *DSF, float *DTF);

void optimised_surfaceKernel_restruct(Mesh *mesh, float *Q2, float *partQ, float *rhsQ2, float *mapinfo, float *surfinfo, float *LIFT, float *kFlux, int *Q_to_Q2);