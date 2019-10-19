#include <math.h>
#include "mpi.h"
#include "fem.h"

int main(int argc, char **argv){

  Mesh *mesh;
  int procid, nprocs, maxNv;
  int k,n, sk=0;
  double minEy, maxEy, gminEy, gmaxEy, gmaxErrorEy;

  /* initialize MPI */
  MPI_Init(&argc, &argv);

  /* assign gpu */
  MPI_Comm_rank(MPI_COMM_WORLD, &procid);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  printf("procid=%d , nprocs=%d\n", procid, nprocs);

#if 0
  /* nicely stop MPI */
    MPI_Finalize();

  /* end game */
  exit(0);
#endif

  /* (parallel) read part of fem mesh from file */
  mesh = ReadMesh3d(argv[1]);

  /* perform load balancing */
  LoadBalance3d(mesh);

  /* find element-element connectivity */
  FacePair3d(mesh, &maxNv);

  /* perform start up */
  StartUp3d(mesh);

  /* field storage (double) */
  double *Hx = (double*) calloc(mesh->K*p_Np, sizeof(double));
  double *Hy = (double*) calloc(mesh->K*p_Np, sizeof(double));
  double *Hz = (double*) calloc(mesh->K*p_Np, sizeof(double));
  double *Ex = (double*) calloc(mesh->K*p_Np, sizeof(double));
  double *Ey = (double*) calloc(mesh->K*p_Np, sizeof(double));
  double *Ez = (double*) calloc(mesh->K*p_Np, sizeof(double));

  /* initial conditions */
  for(k=0;k<mesh->K;++k){
    for(n=0;n<p_Np;++n) {
      Hx[sk] = 0;
      Hy[sk] = 0;
      Hz[sk] = 0;
      Ex[sk] = 0;
      Ey[sk] = sin(M_PI*mesh->x[k][n])*sin(M_PI*mesh->z[k][n]);
      Ez[sk] = 0;
      ++sk;
    }
  }

  double dt, gdt;

  double drdx, dsdx, dtdx;
  double drdy, dsdy, dtdy;
  double drdz, dsdz, dtdz, J;

  double *nxk = BuildVector(mesh->Nfaces);
  double *nyk = BuildVector(mesh->Nfaces);
  double *nzk = BuildVector(mesh->Nfaces);
  double *sJk = BuildVector(mesh->Nfaces);

  float *vgeo = (float*) calloc(9*mesh->K, sizeof(float));
  float *mapinfo = (float*) calloc(2*mesh->K*p_Nfp*p_Nfaces, sizeof(float));
  float *surfinfo = (float*) calloc(5*mesh->K*p_Nfp*p_Nfaces, sizeof(float));

  dt = 1e6;

  for(int k=0;k<mesh->K;++k){

    GeometricFactors3d(mesh, k, 
            &drdx, &dsdx, &dtdx,
            &drdy, &dsdy, &dtdy,
            &drdz, &dsdz, &dtdz, &J);

    vgeo[k*9+0] = drdx; vgeo[k*9+1] = drdy; vgeo[k*9+2] = drdz;
    vgeo[k*9+3] = dsdx; vgeo[k*9+4] = dsdy; vgeo[k*9+5] = dsdz;
    vgeo[k*9+6] = dtdx; vgeo[k*9+7] = dtdy; vgeo[k*9+8] = dtdz;

    Normals3d(mesh, k, nxk, nyk, nzk, sJk); // calculates the cartesian normal vector for each of the 4 faces of an element
      
    for(int f=0; f<mesh->Nfaces; ++f){

      dt = min(dt, J/sJk[f]); // reasonable estimate of the maximum timestep to ensure stable computation
        
      for(int m=0; m<p_Nfp; ++m){
        int n = m + f*p_Nfp + p_Nfp*p_Nfaces*k;
        int idM = mesh->vmapM[n];
        int idP = mesh->vmapP[n];
        int  nM = idM%p_Np; 
        int  nP = idP%p_Np; 
        int  kM = (idM-nM)/p_Np;
        int  kP = (idP-nP)/p_Np;
        idM = nM + p_Nfields*p_Np*kM;
        idP = nP + p_Nfields*p_Np*kP;
  
        /* stub resolve some other way */
        if(mesh->vmapP[n]<0)
          idP = mesh->vmapP[n]; /* -ve numbers */
      
        sk = 2*p_Nfp*p_Nfaces*k+f*p_Nfp+m;
        mapinfo[sk + 0*p_Nfp*p_Nfaces] = idM;
        mapinfo[sk + 1*p_Nfp*p_Nfaces] = idP;

        sk = 5*p_Nfp*p_Nfaces*k+f*p_Nfp+m;
        surfinfo[sk + 0*p_Nfp*p_Nfaces] = sJk[f]/(2.*J);
        surfinfo[sk + 1*p_Nfp*p_Nfaces] = (idM==idP)?-1.:1.;
        surfinfo[sk + 2*p_Nfp*p_Nfaces] = nxk[f];
        surfinfo[sk + 3*p_Nfp*p_Nfaces] = nyk[f];
        surfinfo[sk + 4*p_Nfp*p_Nfaces] = nzk[f];
      }
    }
  }

  free(nxk); 
  free(nyk); 
  free(nzk);
  free(sJk);


  MPI_Allreduce(&dt, &gdt, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);

  dt = .5*gdt/((p_N+1)*(p_N+1));

  //  if(mesh->procid==0)
    printf("dt = %f\n", dt);

    double FinalTime = .75;
  printf("FinalTime=%g\n", FinalTime);

  MaxwellsRun3d(mesh, FinalTime, dt, Hx, Hy, Hz, Ex, Ey, Ez, vgeo, mapinfo, surfinfo);

  free(vgeo);
  free(mapinfo);
  free(surfinfo);


  /* output final field values from each process to a file for checking */
  int token;

  for (int p = 0; p < nprocs; ++p) {

    if (p == procid) {
      
      if (procid != 0) 
        MPI_Recv(&token, 1, MPI_INT, p-1, 420, MPI_COMM_WORLD, MPI_STATUS_IGNORE); 

      FILE* file;

      if (procid == 0)
        file = fopen("/Users/kevintchang/Desktop/Compare/testFinalQ.txt", "w+");
      else
        file = fopen("/Users/kevintchang/Desktop/Compare/testFinalQ.txt", "a");

      for (int k = 0; k < mesh->K; ++k) {
        for (n = 0; n < p_Np; ++n) {
          fprintf(file, "%11f ", Hx[k*p_Np + n]);
        }
        for (n = 0; n < p_Np; ++n) {
          fprintf(file, "%11f ", Hy[k*p_Np + n]);
        }
        for (n = 0; n < p_Np; ++n) {
          fprintf(file, "%11f ", Hz[k*p_Np + n]);
        }
        for (n = 0; n < p_Np; ++n) {
          fprintf(file, "%11f ", Ex[k*p_Np + n]);
        }
        for (n = 0; n < p_Np; ++n) {
          fprintf(file, "%11f ", Ey[k*p_Np + n]);
        }
        for (n = 0; n < p_Np; ++n) {
          fprintf(file, "%11f ", Ez[k*p_Np + n]);
        }
        fputs("\n", file);
      }
      fclose(file);

      if (p < nprocs-1)
        MPI_Send(&token, 1, MPI_INT, p+1, 420, MPI_COMM_WORLD);
    }

  }


  /* find maximum & minimum values for Ez */
  minEy=Ey[0], maxEy=Ey[0];

  double maxErrorEy = 0;
  for(k=0;k<mesh->K;++k) {
    for(n=0;n<p_Np;++n){
      int id = n + p_Np*k;
      double exactEy = sin(M_PI*mesh->x[k][n])*sin(M_PI*mesh->z[k][n])*cos(sqrt(2.)*M_PI*FinalTime);
      double errorEy = fabs(exactEy-Ey[id]);
      maxErrorEy = (errorEy>maxErrorEy) ? errorEy:maxErrorEy;
      minEy = (minEy>Ey[id])?Ey[id]:minEy;
      maxEy = (maxEy<Ey[id])?Ey[id]:maxEy;
    }
  }

  MPI_Reduce(&minEy, &gminEy, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
  MPI_Reduce(&maxEy, &gmaxEy, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(&maxErrorEy, &gmaxErrorEy, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

  if(procid==0)
    printf("t=%f Ey in [ %g, %g ] with max nodal error %g \n", FinalTime, gminEy, gmaxEy, gmaxErrorEy );

  /* nicely stop MPI */
  MPI_Finalize();

  /* end game */
  exit(0);
}
