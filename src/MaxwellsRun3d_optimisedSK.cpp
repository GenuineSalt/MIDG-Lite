#include "mpi.h"
#include "fem.h"

static MPI_Request *mpi_out_requests = NULL;
static MPI_Request *mpi_in_requests  = NULL;

static int Nmess = 0;

void MaxwellsRun3d(Mesh *mesh, double FinalTime, double dt, double *Hx, double *Hy, double *Hz, double *Ex, double *Ey, double *Ez, float *vgeo, float *mapinfo, float *surfinfo){

    /* field data memory */
    float* Q = (float*) calloc(mesh->K*p_Np*p_Nfields, sizeof(float));
    float* rhsQ = (float*) calloc(mesh->K*p_Np*p_Nfields, sizeof(float));
    float* resQ = (float*) calloc(mesh->K*p_Np*p_Nfields, sizeof(float));

    float* partQ = (float*) calloc(mesh->parNtotalout, sizeof(float));
 
    for(int k=0;k<mesh->K;++k){
        for(int n=0;n<p_Np;++n)
            Q[n       +k*p_Np*p_Nfields] = Hx[n+k*p_Np];
        for(int n=0;n<p_Np;++n)
            Q[n  +p_Np+k*p_Np*p_Nfields] = Hy[n+k*p_Np];
        for(int n=0;n<p_Np;++n)
            Q[n+2*p_Np+k*p_Np*p_Nfields] = Hz[n+k*p_Np];
        for(int n=0;n<p_Np;++n)
            Q[n+3*p_Np+k*p_Np*p_Nfields] = Ex[n+k*p_Np];
        for(int n=0;n<p_Np;++n)
            Q[n+4*p_Np+k*p_Np*p_Nfields] = Ey[n+k*p_Np];
        for(int n=0;n<p_Np;++n)
            Q[n+5*p_Np+k*p_Np*p_Nfields] = Ez[n+k*p_Np];
    }

    /* auxiliary data memory */
    float* LIFT = (float*) calloc(p_Np*p_Nfp*p_Nfaces, sizeof(float));

    for(int n=0; n<p_Np; ++n){
        for(int m=0; m<p_Nfp*p_Nfaces; ++m){
            LIFT[n*p_Nfp*p_Nfaces + m] = mesh->LIFT[n][m];
        }
    }

    /* optimisation matrices */
    float* kFlux = (float*) calloc(p_Nfields*p_Nfaces*p_Nfp, sizeof(float));
    float* kField = (float*) calloc(p_Nfields*p_Np, sizeof(float));


    double time = 0;
    int    INTRK, tstep=0;

    MPI_Barrier(MPI_COMM_WORLD);
    double mpitime0 = MPI_Wtime();  // start measuring wall-clock time
    double cpu0 = (double)clock() / CLOCKS_PER_SEC; // start measuring cpu time

    int Nsteps = FinalTime/dt;
    dt = FinalTime/Nsteps;

    /* outer time step loop  */
    for(tstep=0;tstep<Nsteps;++tstep){

        for (INTRK=1; INTRK<=5; ++INTRK) {  

            const float fdt = dt;
            const float fa = (float)mesh->rk4a[INTRK-1];
            const float fb = (float)mesh->rk4b[INTRK-1];

            MaxwellsMPISend3d(mesh, Q);

            volumeKernel(mesh, Q, rhsQ, vgeo);

            MaxwellsMPIRecv3d(mesh, partQ);

            optimised_surfaceKernel(mesh, Q, partQ, rhsQ, mapinfo, surfinfo, LIFT, kFlux, kField);

            rkKernel(mesh, resQ, rhsQ, Q, fa, fb, fdt);
        }

        time += dt;     /* increment current time */
    }

    double mpitime1 = MPI_Wtime();  // stop measuring wall-clock time
    double cpu1 = (double)clock() / CLOCKS_PER_SEC; // stop measuring cpu time
    double wall_time_total = mpitime1-mpitime0;
    double cpu_time_total = cpu1-cpu0;

    MPI_Barrier(MPI_COMM_WORLD);
    printf("proc: %d, p_N: %d, time taken (wall-clock): %lg, time taken (cpu): %lg\n",
    mesh->procid,
    p_N, 
    wall_time_total,
    cpu_time_total);
    

    /* recover field values */
    for(int k=0; k<mesh->K; ++k) {
        for(int n=0;n<p_Np;++n)
            Hx[n+k*p_Np] = Q[n        +k*BSIZE*p_Nfields];
        for(int n=0;n<p_Np;++n) 
            Hy[n+k*p_Np] = Q[n  +BSIZE+k*BSIZE*p_Nfields];
        for(int n=0;n<p_Np;++n)
            Hz[n+k*p_Np] = Q[n+2*BSIZE+k*BSIZE*p_Nfields];
        for(int n=0;n<p_Np;++n)
            Ex[n+k*p_Np] = Q[n+3*BSIZE+k*BSIZE*p_Nfields];
        for(int n=0;n<p_Np;++n) 
            Ey[n+k*p_Np] = Q[n+4*BSIZE+k*BSIZE*p_Nfields];
        for(int n=0;n<p_Np;++n)
            Ez[n+k*p_Np] = Q[n+5*BSIZE+k*BSIZE*p_Nfields];
    }

    free(Q);
    free(rhsQ);
    free(resQ);
    free(partQ);

    free(LIFT);

    free(kFlux);
    free(kField);
}



void MaxwellsMPISend3d(Mesh *mesh, float *Q) {

    int p;

    int procid = mesh->procid;
    int nprocs = mesh->nprocs;

    MPI_Status status;

    if(mpi_out_requests==NULL){
        mpi_out_requests = (MPI_Request*) calloc(nprocs, sizeof(MPI_Request));
        mpi_in_requests  = (MPI_Request*) calloc(nprocs, sizeof(MPI_Request));
    }

    if(mesh->parNtotalout){

        for (int n = 0; n < mesh->parNtotalout; ++n) {
            mesh->f_outQ[n] = Q[mesh->parmapOUT[n]];
        }
    }

    /* non-blocked send/recv partition surface data */
    Nmess = 0;

    /* now send piece to each proc */
    int sk = 0;
    for(p=0;p<nprocs;++p){

        if(p!=procid){
            int Nout = mesh->Npar[p]*p_Nfields*p_Nfp;
            if(Nout){
                /* symmetric communications (different ordering) */
                MPI_Isend(mesh->f_outQ+sk, Nout, MPI_FLOAT, p, 6666+p,      MPI_COMM_WORLD, mpi_out_requests +Nmess);
                MPI_Irecv(mesh->f_inQ+sk,  Nout, MPI_FLOAT, p, 6666+procid, MPI_COMM_WORLD,  mpi_in_requests +Nmess);
                sk+=Nout;
                ++Nmess;
            }
        }
    }
}



void MaxwellsMPIRecv3d(Mesh *mesh, float *partQ){
    int p, n;
    int nprocs = mesh->nprocs;

    MPI_Status *instatus  = (MPI_Status*) calloc(nprocs, sizeof(MPI_Status));
    MPI_Status *outstatus = (MPI_Status*) calloc(nprocs, sizeof(MPI_Status));

    MPI_Waitall(Nmess, mpi_in_requests, instatus);

    if(mesh->parNtotalout) {

        for (n = 0; n < mesh->parNtotalout; ++n) {
            partQ[n] = mesh->f_inQ[n];
        } 
    }

    MPI_Waitall(Nmess, mpi_out_requests, outstatus);

    free(outstatus);
    free(instatus);
}



void volumeKernel(Mesh *mesh, float *Q, float *rhsQ, float *vgeo) {

    for (int k = 0; k < mesh->K; ++k) {

        float Hx[p_Np];
        float Hy[p_Np];
        float Hz[p_Np];
        float Ex[p_Np];
        float Ey[p_Np];
        float Ez[p_Np];

        for (int n = 0; n < p_Np; ++n) {
            Hx[n] = Q[k*p_Nfields*p_Np + n];
            Hy[n] = Q[k*p_Nfields*p_Np + n + p_Np];
            Hz[n] = Q[k*p_Nfields*p_Np + n + 2*p_Np];
            Ex[n] = Q[k*p_Nfields*p_Np + n + 3*p_Np];
            Ey[n] = Q[k*p_Nfields*p_Np + n + 4*p_Np];
            Ez[n] = Q[k*p_Nfields*p_Np + n + 5*p_Np];
        }

        for (int n = 0; n < p_Np; ++n) {

            float dHxdr=0,dHxds=0,dHxdt=0;
            float dHydr=0,dHyds=0,dHydt=0;
            float dHzdr=0,dHzds=0,dHzdt=0;
            float dExdr=0,dExds=0,dExdt=0;
            float dEydr=0,dEyds=0,dEydt=0;
            float dEzdr=0,dEzds=0,dEzdt=0;

            for (int m = 0; m < p_Np; ++m) {
                
                float Dr = mesh->Dr[n][m];
                float Ds = mesh->Ds[n][m];
                float Dt = mesh->Dt[n][m];

                dHxdr += Dr*Hx[m]; dHxds += Ds*Hx[m]; dHxdt += Dt*Hx[m];
                dHydr += Dr*Hy[m]; dHyds += Ds*Hy[m]; dHydt += Dt*Hy[m];
                dHzdr += Dr*Hz[m]; dHzds += Ds*Hz[m]; dHzdt += Dt*Hz[m];
                dExdr += Dr*Ex[m]; dExds += Ds*Ex[m]; dExdt += Dt*Ex[m];
                dEydr += Dr*Ey[m]; dEyds += Ds*Ey[m]; dEydt += Dt*Ey[m];
                dEzdr += Dr*Ez[m]; dEzds += Ds*Ez[m]; dEzdt += Dt*Ez[m];
            }

            float drdx = vgeo[9*k];
            float drdy = vgeo[9*k + 1];
            float drdz = vgeo[9*k + 2];
            float dsdx = vgeo[9*k + 3];
            float dsdy = vgeo[9*k + 4];
            float dsdz = vgeo[9*k + 5];
            float dtdx = vgeo[9*k + 6];
            float dtdy = vgeo[9*k + 7];
            float dtdz = vgeo[9*k + 8];

            int m = k*p_Nfields*p_Np + n;

            rhsQ[m] = -(drdy*dEzdr+dsdy*dEzds+dtdy*dEzdt - drdz*dEydr-dsdz*dEyds-dtdz*dEydt); m += p_Np;
            rhsQ[m] = -(drdz*dExdr+dsdz*dExds+dtdz*dExdt - drdx*dEzdr-dsdx*dEzds-dtdx*dEzdt); m += p_Np;
            rhsQ[m] = -(drdx*dEydr+dsdx*dEyds+dtdx*dEydt - drdy*dExdr-dsdy*dExds-dtdy*dExdt); m += p_Np;
            rhsQ[m] =  (drdy*dHzdr+dsdy*dHzds+dtdy*dHzdt - drdz*dHydr-dsdz*dHyds-dtdz*dHydt); m += p_Np;
            rhsQ[m] =  (drdz*dHxdr+dsdz*dHxds+dtdz*dHxdt - drdx*dHzdr-dsdx*dHzds-dtdx*dHzdt); m += p_Np;
            rhsQ[m] =  (drdx*dHydr+dsdx*dHyds+dtdx*dHydt - drdy*dHxdr-dsdy*dHxds-dtdy*dHxdt); 
        }
    }
}



void optimised_surfaceKernel(Mesh *mesh, float *Q, float *partQ, float *rhsQ, float *mapinfo, float *surfinfo, float *LIFT, float *kFlux, float *kField) {

    float dHx, dHy, dHz, dEx, dEy, dEz;

    for (int k = 0; k < mesh->K; ++k) {

        /* retrieve surface node data and calculate flux (coalesced reads) */
        for (int m = 0; m < p_Nfaces*p_Nfp; ++m) {

            int imap = 2*k*p_Nfaces*p_Nfp + m;
            int isurf = 5*k*p_Nfaces*p_Nfp + m;

            int idM = mapinfo[imap]; imap +=p_Nfaces*p_Nfp;
            int idP = mapinfo[imap];

            float Fsc = surfinfo[isurf]; isurf += p_Nfaces*p_Nfp;
            float Bsc = surfinfo[isurf]; isurf += p_Nfaces*p_Nfp;
            float nx = surfinfo[isurf]; isurf += p_Nfaces*p_Nfp;
            float ny = surfinfo[isurf]; isurf += p_Nfaces*p_Nfp;
            float nz = surfinfo[isurf]; 
            
            
            if (idP < 0) {
                idP = p_Nfields*(-1-idP);
				
				dHx = Fsc*(partQ[idP+0] - Q[idM+0*p_Np]);
				dHy = Fsc*(partQ[idP+1] - Q[idM+1*p_Np]);
				dHz = Fsc*(partQ[idP+2] - Q[idM+2*p_Np]);
				dEx = Fsc*(partQ[idP+3] - Q[idM+3*p_Np]);
				dEy = Fsc*(partQ[idP+4] - Q[idM+4*p_Np]);
				dEz = Fsc*(partQ[idP+5] - Q[idM+5*p_Np]);
            }

            else {
                dHx = Fsc*(Q[idP+0*p_Np] - Q[idM+0*p_Np]);
				dHy = Fsc*(Q[idP+1*p_Np] - Q[idM+1*p_Np]);
				dHz = Fsc*(Q[idP+2*p_Np] - Q[idM+2*p_Np]);
				dEx = Fsc*(Bsc*Q[idP+3*p_Np] - Q[idM+3*p_Np]);
				dEy = Fsc*(Bsc*Q[idP+4*p_Np] - Q[idM+4*p_Np]);
				dEz = Fsc*(Bsc*Q[idP+5*p_Np] - Q[idM+5*p_Np]);
            }

            float ndotdH = nx*dHx + ny*dHy + nz*dHz;
            float ndotdE = nx*dEx + ny*dEy + nz*dEz;

            kFlux[6*m] = -ny*dEz + nz*dEy + dHx - ndotdH*nx;
            kFlux[6*m + 1] = -nz*dEx + nx*dEz + dHy - ndotdH*ny;
            kFlux[6*m + 2] = -nx*dEy + ny*dEx + dHz - ndotdH*nz;
            kFlux[6*m + 3] = ny*dHz - nz*dHy + dEx - ndotdE*nx;
            kFlux[6*m + 4] = nz*dHx - nx*dHz + dEy - ndotdE*ny;
            kFlux[6*m + 5] = nx*dHy - ny*dHx + dEz - ndotdE*nz;
        }

        /* Matrix multiply LIFT with kFlux */
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, p_Np, p_Nfields, p_Nfaces*p_Nfp, 1, (float*)LIFT, p_Nfaces*p_Nfp, (float*)kFlux, p_Nfields, 0, (float*)kField, p_Nfields);

        for (int n = 0; n < p_Np; ++n) {

            int m = k*p_Nfields*p_Np + n;

            rhsQ[m] += kField[6*n]; m += p_Np;
            rhsQ[m] += kField[6*n + 1]; m += p_Np;
            rhsQ[m] += kField[6*n + 2]; m += p_Np;
            rhsQ[m] += kField[6*n + 3]; m += p_Np;
            rhsQ[m] += kField[6*n + 4]; m += p_Np;
            rhsQ[m] += kField[6*n + 5];
        }
    }
}



void rkKernel(Mesh *mesh, float *resQ, float *rhsQ, float *Q, float fa, float fb, float fdt) {

    for (int n = 0; n < mesh->K*p_Nfields*p_Np; ++n) {

        resQ[n] = fa*resQ[n] + fdt*rhsQ[n];

        Q[n] = Q[n] + fb*resQ[n];
    }
}