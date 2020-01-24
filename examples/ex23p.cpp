//                       MFEM Example 23 - Parallel Version
//
// Compile with: make ex23p
//
// Sample runs:  mpirun -np 4 ex23p -m ../../data/square-disc.mesh -o 2
//               mpirun -np 4 ex23p -m ../../data/beam-tet.mesh
//               mpirun -np 4 ex23p -m ../../data/beam-hex.mesh
//               mpirun -np 4 ex23p -m ../../data/fichera.mesh
//               mpirun -np 4 ex23p -m ../../data/amr-quad.mesh -o 2
//               mpirun -np 4 ex23p -m ../../data/amr-hex.mesh
//               mpirun -np 4 ex23p -m ../../hexa728.mesh
//               mpirun -np 4 ex23p -m ../../data/rectwhole7_2attr.e
// Description:  This example code solves a simple electromagnetic wave
//               propagation problem corresponding to the second order indefinite
//               Maxwell equation curl curl E - \omega^2 E = f with a PML
//               We discretize with Nedelec finite elements in 2D or 3D.
//
//               The example also demonstrates the use of complex valued bilinear and linear forms.
//               We recommend viewing example 22 before viewing this example.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

#ifdef _WIN32
#define jn(n, x) _jn(n, x)
#endif

using namespace std;
using namespace mfem;

#ifndef MFEM_USE_SUPERLU
#error This example requires that MFEM is built with MFEM_USE_SUPERLU=YES
#endif

// Exact solution, E, and r.h.s., f. See below for implementation.
// void compute_pml_mesh_data(Mesh *mesh);
void maxwell_ess_data(const Vector &x, std::vector<std::complex<double>> &Eval);
void E_bdr_data_Re(const Vector &x, Vector &E);
void E_bdr_data_Im(const Vector &x, Vector &E);
void E_exact_Re(const Vector &x, Vector &E);
void E_exact_Im(const Vector &x, Vector &E);

void source(const Vector &x, Vector & f);
double pml_detJ_inv_Re(const Vector &x);
double pml_detJ_inv_Im(const Vector &x);
void pml_detJ_JT_J_inv_Re(const Vector &x, DenseMatrix &M);
void pml_detJ_JT_J_inv_Im(const Vector &x, DenseMatrix &M);
void pml_detJ_inv_JT_J_Re(const Vector &x, DenseMatrix &M);
void pml_detJ_inv_JT_J_Im(const Vector &x, DenseMatrix &M);

class pml
{
private:
   Mesh *mesh;
   int dim;

public:
   Array2D<double> domain_bdr;
   Array2D<double> comp_domain_bdr;
   Array2D<double> length;
   Array<int> elems;
   pml() {};
   pml(Mesh *mesh_);

   // Set the pml length in each dimension. Size should be (dim,2)
   void compute_data();
   void SetMesh(Mesh *mesh_);
   void SetPmlLength(Array2D<double> &length_);
   void get_pml_elem_list(ParMesh *pmesh, Array<int> &elem_pml);
};

pml p;
double omega;
int dim;

enum prob_type
{
   load_src,
   scatter,
   lshape,
   fichera,
   waveguide
};
prob_type prob = scatter;

int main(int argc, char *argv[])
{
   // 1. Initialize MPI.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   // 2. Parse command-line options.
   const char *mesh_file = "../data/inline-quad.mesh";
   int order = 1;
   bool visualization = 1;
   double freq = 1.0;
   int ref_levels = 1;
   int par_ref_levels = 1;
   int iprob = 0;
   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&freq, "-f", "--frequency", "Set the frequency for the exact"
                  " solution.");
   args.AddOption(&iprob, "-prob", "--problem", "Problem case"
                  " 0: General, 1: scatter, 2: lshape, 3: fichera, 4: waveguide, 5: Circular waveguide.");
   args.AddOption(&ref_levels, "-rs", "--refinements-serial",
                  "Number of serial refinements");
   args.AddOption(&par_ref_levels, "-rp", "--refinements-parallel",
                  "Number of parallel refinements");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();
   if (!args.Good())
   {
      if (myid == 0)
      {
         args.PrintUsage(cout);
      }
      MPI_Finalize();
      return 1;
   }
   if (myid == 0)
   {
      args.PrintOptions(cout);
   }
   omega = 2.0 * M_PI * freq;

   // 3. Read the (serial) mesh from the given mesh file on all processors.
   Mesh *mesh;
   if (iprob > 4)
   {
      iprob = 0;
   }
   prob = (prob_type)iprob;

   switch (prob)
   {
      case scatter:
         mesh_file = "../data/square_w_hole.mesh";
         break;
      case lshape:
         mesh_file = "l-shape.mesh";
         break;
      case fichera:
         mesh_file = "fichera.mesh";
         break;
      case waveguide:
         mesh_file = "../data/beam-hex.mesh";
         break;
      default:
         break;
   }

   mesh = new Mesh(mesh_file, 1, 1);
   dim = mesh->Dimension();
   Array2D<double> lngth(dim, 2);
   lngth = 0.0;

   switch (prob)
   {
      case scatter:
         lngth = 1.0;
         break;
      case lshape:
         lngth(0, 1) = 0.5;
         lngth(1, 1) = 0.5;
         break;
      case fichera:
         lngth(0, 1) = 0.5;
         lngth(1, 1) = 0.5;
         lngth(2, 1) = 0.5;
         break;
      case waveguide:
         lngth(0, 1) = 1.0;
         break;
      default:
         lngth = 0.25;
         break;
   }

   p.SetMesh(mesh);
   p.SetPmlLength(lngth);
   p.compute_data();

   // 4. Refine the serial mesh on all processors to increase the resolution.
   for (int l = 0; l < ref_levels; l++)
   {
      mesh->UniformRefinement();
   }

   // 5. Define a parallel mesh by a partitioning of the serial mesh.
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   {
      for (int l = 0; l < par_ref_levels; l++)
      {
         pmesh->UniformRefinement();
      }
   }

   pmesh->ReorientTetMesh();
   Array<int> elems_pml;
   p.get_pml_elem_list(pmesh, elems_pml);

   // 6. Define a parallel finite element space on the parallel mesh. Here we
   //    use the Nedelec finite elements of the specified order.
   FiniteElementCollection *fec = new ND_FECollection(order, dim);
   ParFiniteElementSpace *fespace = new ParFiniteElementSpace(pmesh, fec);
   HYPRE_Int size = fespace->GlobalTrueVSize();
   if (myid == 0)
   {
      cout << "Number of finite element unknowns: " << size << endl;
   }

   // 7. Determine the list of true (i.e. parallel conforming) essential
   //    boundary dofs.
   Array<int> ess_tdof_list;
   Array<int> ess_bdr;
   if (pmesh->bdr_attributes.Size())
   {
      ess_bdr.SetSize(pmesh->bdr_attributes.Max());
      ess_bdr = 0;
      switch (prob)
      {
         case lshape:
            for (int j = 0; j < pmesh->GetNBE(); j++)
            {
               Array<int> vertices;
               pmesh->GetBdrElementVertices(j, vertices);
               double *coords1 = pmesh->GetVertex(vertices[0]);
               double *coords2 = pmesh->GetVertex(vertices[1]);
               int k = pmesh->GetBdrAttribute(j);
               if ((coords1[0] == 1.0 && coords2[0] == 1.0) ||
                   (coords1[0] == 0.0 && coords2[0] == 0.0) ||
                   (coords1[1] == 1.0 && coords2[1] == 1.0))
               {

                  ess_bdr[k - 1] = 1;
               }
            }
            break;
         case fichera:
            for (int j = 0; j < pmesh->GetNBE(); j++)
            {
               Array<int> vertices;
               pmesh->GetBdrElementVertices(j, vertices);
               double *coords1 = pmesh->GetVertex(vertices[0]);
               double *coords2 = pmesh->GetVertex(vertices[1]);
               double *coords3 = pmesh->GetVertex(vertices[2]);
               double *coords4 = pmesh->GetVertex(vertices[3]);
               int k = pmesh->GetBdrAttribute(j);

               if ((coords1[0] == -1.0 && coords2[0] == -1.0 && coords3[0] == -1.0 &&
                    coords4[0] == -1.0) ||
                   (coords1[0] == 0.0 && coords2[0] == 0.0 && coords3[0] == 0.0 &&
                    coords4[0] == 0.0) ||
                   (coords1[1] == 0.0 && coords2[1] == 0.0 && coords3[1] == 0.0 &&
                    coords4[1] == 0.0) ||
                   (coords1[2] == 0.0 && coords2[2] == 0.0 && coords3[2] == 0.0 &&
                    coords4[2] == 0.0))
               {
                  ess_bdr[k - 1] = 1;
               }
            }
            break;
         default:
            ess_bdr = 1;
            break;
      }
      fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   // 8. Set up the parallel linear form b(.)
   VectorFunctionCoefficient f(dim, source);
   ParComplexLinearForm b(fespace, ComplexOperator::HERMITIAN);
   if (prob == load_src)
   {
      b.AddDomainIntegrator(NULL, new VectorFEDomainLFIntegrator(f));
   }
   b.real().Vector::operator=(0.0);
   b.imag().Vector::operator=(0.0);
   b.Assemble();

   // 9. Define the solution vector x
   ParComplexGridFunction x(fespace);
   VectorFunctionCoefficient E_Re(dim, E_bdr_data_Re);
   VectorFunctionCoefficient E_Im(dim, E_bdr_data_Im);
   x.ProjectBdrCoefficientTangent(E_Re, E_Im, ess_bdr);

   // 10. Set up the parallel bilinear form
   ConstantCoefficient muinv(1.0);
   ConstantCoefficient sigma(-pow(omega, 2));
   FunctionCoefficient det_inv_Re(pml_detJ_inv_Re);
   FunctionCoefficient det_inv_Im(pml_detJ_inv_Im);
   MatrixFunctionCoefficient temp_c1_Re(dim, pml_detJ_inv_JT_J_Re);
   MatrixFunctionCoefficient temp_c1_Im(dim, pml_detJ_inv_JT_J_Im);
   MatrixFunctionCoefficient temp_c2_Re(dim, pml_detJ_JT_J_inv_Re);
   MatrixFunctionCoefficient temp_c2_Im(dim, pml_detJ_JT_J_inv_Im);

   ScalarMatrixProductCoefficient pml_c1_Re(muinv, temp_c1_Re);
   ScalarMatrixProductCoefficient pml_c1_Im(muinv, temp_c1_Im);
   ScalarMatrixProductCoefficient pml_c2_Re(sigma, temp_c2_Re);
   ScalarMatrixProductCoefficient pml_c2_Im(sigma, temp_c2_Im);

   ParSesquilinearForm a(fespace, ComplexOperator::HERMITIAN);
   if (dim == 3)
   {
      a.AddDomainIntegrator(new CurlCurlIntegrator(pml_c1_Re),
                            new CurlCurlIntegrator(pml_c1_Im));
   }
   else
   {
      a.AddDomainIntegrator(new CurlCurlIntegrator(det_inv_Re),
                            new CurlCurlIntegrator(det_inv_Im));
   }
   a.AddDomainIntegrator(new VectorFEMassIntegrator(pml_c2_Re),
                         new VectorFEMassIntegrator(pml_c2_Im));
   a.Assemble();

   OperatorHandle Ah;
   Vector B, X;
   a.FormLinearSystem(ess_tdof_list, x, b, Ah, X, B);

   // Transform to monolithic HypreParMatrix
   HypreParMatrix *A = Ah.As<ComplexHypreParMatrix>()->GetSystemMatrix();

   if (myid == 0)
   {
      cout << "Size of linear system: " << A->GetGlobalNumRows() << endl;
   }

   // SuperLU direct solver
   SuperLURowLocMatrix *SA = new SuperLURowLocMatrix(*A);
   SuperLUSolver *superlu = new SuperLUSolver(MPI_COMM_WORLD);
   superlu->SetPrintStatistics(false);
   superlu->SetSymmetricPattern(false);
   superlu->SetColumnPermutation(superlu::PARMETIS);
   superlu->SetOperator(*SA);
   superlu->Mult(B, X);

   // 13. Recover the parallel grid function corresponding to X. This is the
   //     local finite element solution on each processor.
   a.RecoverFEMSolution(X, b, x);


   // Compute error
   if (prob == scatter || prob == waveguide || prob == lshape || prob == fichera)
   {
      ParComplexGridFunction x_gf(fespace);
      VectorFunctionCoefficient E_ex_Re(dim, E_exact_Re);
      VectorFunctionCoefficient E_ex_Im(dim, E_exact_Im);
      x_gf.ProjectCoefficient(E_ex_Re, E_ex_Im);
      int order_quad = max(2, 2 * order + 1);
      const IntegrationRule *irs[Geometry::NumGeom];
      for (int i = 0; i < Geometry::NumGeom; ++i)
      {
         irs[i] = &(IntRules.Get(i, order_quad));
      }

      double L2Error_Re = x.real().ComputeL2Error(E_ex_Re, irs, &elems_pml);
      double L2Error_Im = x.imag().ComputeL2Error(E_ex_Im, irs, &elems_pml);

      ParComplexGridFunction x_gf0(fespace);
      x_gf0 = 0.0;
      double norm_E_Re = x_gf0.real().ComputeL2Error(E_ex_Re, irs, &elems_pml);
      double norm_E_Im = x_gf0.imag().ComputeL2Error(E_ex_Im, irs, &elems_pml);

      if (myid == 0)
      {
         cout << " Rel Error - Real Part: || E_h - E || / ||E|| = " << L2Error_Re /
              norm_E_Re << '\n'
              << endl;
         cout << " Rel Error - Imag Part: || E_h - E || / ||E|| = " << L2Error_Im /
              norm_E_Im << '\n'
              << endl;
         cout << " Total Error: " << sqrt(L2Error_Re * L2Error_Re + L2Error_Im *
                                          L2Error_Im)
              << endl;
      }
   }

   // 16. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      string keys;
      if (dim == 3)
      {
         keys = "keys mF\n";
      }
      else
      {
         keys = "keys arRljcUU\n";
      }
      char vishost[] = "localhost";
      int visport = 19916;

      MPI_Barrier(MPI_COMM_WORLD);
      socketstream sol_sock_re(vishost, visport);
      sol_sock_re << "parallel " << num_procs << " " << myid << "\n";
      sol_sock_re.precision(8);
      sol_sock_re << "solution\n"
                  << *pmesh << x.real() << keys
                  << "window_title 'Solution real part'" << flush;

      MPI_Barrier(MPI_COMM_WORLD);
      socketstream sol_sock_im(vishost, visport);
      sol_sock_im << "parallel " << num_procs << " " << myid << "\n";
      sol_sock_im.precision(8);
      sol_sock_im << "solution\n"
                  << *pmesh << x.imag() << keys
                  << "window_title 'Solution imag part'" << flush;

      MPI_Barrier(MPI_COMM_WORLD);
      ParGridFunction x_t(fespace);
      x_t = x.real();
      keys = "keys rRljcUUuu\n";
      socketstream sol_sock(vishost, visport);
      sol_sock << "parallel " << num_procs << " " << myid << "\n";
      sol_sock.precision(8);
      sol_sock << "solution\n"
               << *pmesh << x_t << keys << "autoscale off\n"
               // << "valuerange -4.0  4.0 \n "
               << "window_title 'Harmonic Solution (t = 0.0 T)'"
               << "pause\n"
               << flush;
      if (myid == 0)
         cout << "GLVis visualization paused."
              << " Press space (in the GLVis window) to resume it.\n";
      int num_frames = 32;
      int i = 0;
      while (sol_sock)
      {
         double t = (double)(i % num_frames) / num_frames;
         ostringstream oss;
         oss << "Harmonic Solution (t = " << t << " T)";

         add(cos(omega * t), x.real(),
             sin(omega * t), x.imag(), x_t);
         sol_sock << "parallel " << num_procs << " " << myid << "\n";
         sol_sock << "solution\n"
                  << *pmesh << x_t
                  << "window_title '" << oss.str() << "'" << flush;
         i++;
      }
   }

   // 17. Free the used memory.
   delete superlu;
   delete SA;
   delete fespace;
   delete fec;
   delete pmesh;

   MPI_Finalize();
   return 0;
}

void source(const Vector &x, Vector &f)
{
   Vector center(dim);
   double r = 0.0;
   for (int i = 0; i < dim; ++i)
   {
      center(i) = 0.5 * (p.comp_domain_bdr(i, 0) + p.comp_domain_bdr(i, 1));
      r += pow(x[i] - center[i], 2.);
   }
   double n = 5.0 * omega / M_PI;
   double coeff = pow(n, 2) / M_PI;
   double alpha = -pow(n, 2) * r;
   f[0] = coeff * exp(alpha);
}

void maxwell_ess_data(const Vector &x, std::vector<std::complex<double>> &E)
{
   // Initialize
   for (int i = 0; i < dim; ++i) { E[i] = complex<double>(0., 0.); }

   std::complex<double> zi = std::complex<double>(0., 1.);

   switch (prob)
   {
      case scatter:
      case lshape:
      case fichera:
      {
         Vector shift(dim);
         shift = 0.0;
         if (prob == fichera) { shift = 1.0; }

         if (dim == 2)
         {
            double x0 = x(0) + shift(0);
            double x1 = x(1) + shift(1);
            std::complex<double> val, val_x, val_xx, val_xy;
            double r = sqrt(x0 * x0 + x1 * x1);
            double beta = omega * r;

            // Bessel functions
            complex<double> Ho = jn(0, beta) + zi * yn(0, beta);
            complex<double> Ho_r = -omega * (jn(1, beta) + zi * yn(1, beta));
            complex<double> Ho_rr = -omega * omega * (1.0 / beta * (jn(1, beta) + zi * yn(1,
                                                                                          beta)) - (jn(2, beta) + zi * yn(2, beta)));

            // First derivatives
            double r_x = x0 / r;
            double r_y = x1 / r;
            double r_xy = -(r_x / r) * r_y;
            double r_xx = (1.0 / r) * (1.0 - r_x * r_x);

            val = 0.25 * zi * Ho; // i/4 * H_0^1(omega * r)
            val_x = 0.25 * zi * r_x * Ho_r;
            val_xx = 0.25 * zi * (r_xx * Ho_r + r_x * r_x * Ho_rr);
            val_xy = 0.25 * zi * (r_xy * Ho_r + r_x * r_y * Ho_rr);
            E[0] = zi / omega * (omega * omega * val + val_xx);
            E[1] = zi / omega * val_xy;
         }
         else if (dim == 3)
         {
            double x0 = x(0) + shift(0);
            double x1 = x(1) + shift(1);
            double x2 = x(2) + shift(2);
            double r = sqrt(x0 * x0 + x1 * x1 + x2 * x2);

            double r_x = x0 / r;
            double r_y = x1 / r;
            double r_z = x2 / r;
            double r_xx = (1.0 / r) * (1.0 - r_x * r_x);
            double r_yx = -(r_y / r) * r_x;
            double r_zx = -(r_z / r) * r_x;

            complex<double> val, val_x, val_y, val_z;
            complex<double> val_xx, val_yx, val_zx;
            complex<double> val_r, val_rr;

            val = exp(zi * omega * r) / r;
            val_r = val / r * (zi * omega * r - 1.0);
            val_rr = val / (r * r) * (-omega * omega * r * r - 2.0 * zi * omega * r + 2.0);
            val_x = val_r * r_x;
            val_y = val_r * r_y;
            val_z = val_r * r_z;

            val_xx = val_rr * r_x * r_x + val_r * r_xx;
            val_yx = val_rr * r_x * r_y + val_r * r_yx;
            val_zx = val_rr * r_x * r_z + val_r * r_zx;

            complex<double> alpha = zi * omega / 4.0 / M_PI / omega / omega;
            E[0] = alpha * (omega * omega * val + val_xx);
            E[1] = alpha * val_yx;
            E[2] = alpha * val_zx;
         }
         break;
      }
      case waveguide:
      {
         double k10 = sqrt(omega * omega - M_PI * M_PI);
         E[1] = -zi * omega / M_PI * sin(M_PI * x(2)) * exp(zi * k10 * x(
                                                               0)); // T_10 mode
         break;
      }
      default:
         break;
   }
}

void E_exact_Re(const Vector &x, Vector &E)
{
   std::vector<std::complex<double>> Eval(E.Size());
   maxwell_ess_data(x, Eval);
   for (int i = 0; i < dim; ++i)
   {
      E[i] = Eval[i].real();
   }
}

void E_exact_Im(const Vector &x, Vector &E)
{
   std::vector<std::complex<double>> Eval(E.Size());
   maxwell_ess_data(x, Eval);
   for (int i = 0; i < dim; ++i)
   {
      E[i] = Eval[i].imag();
   }
}

void E_bdr_data_Re(const Vector &x, Vector &E)
{
   // Initialize
   E = 0.0;
   bool in_pml = false;
   if (prob == scatter)
   {
      for (int i = 0; i < dim; ++i)
      {
         // check if x(i) is in the computational domain or not
         if (abs(x(i) - p.domain_bdr(i, 0)) < 1e-13 ||
             abs(x(i) - p.domain_bdr(i, 1)) < 1e-13)
         {
            in_pml = true;
            break;
         }
      }
      if (!in_pml)
      {
         std::vector<std::complex<double>> Eval(E.Size());
         maxwell_ess_data(x, Eval);
         for (int i = 0; i < dim; ++i)
         {
            E[i] = Eval[i].real();
         }
      }
   }
   else if (prob == lshape)
   {
      if ((x(0) == 1.0 && x(1) < 1.0) ||
          (x(1) == 1.0 && x(0) < 1.0))
      {
         std::vector<std::complex<double>> Eval(E.Size());
         maxwell_ess_data(x, Eval);
         for (int i = 0; i < dim; ++i)
         {
            E[i] = Eval[i].real();
         }
      }
   }
   else if (prob == fichera)
   {
      if ((x(0) == 0.0 && x(1) < 0.0 && x(2) < 0.0) ||
          (x(1) == 0.0 && x(0) < 0.0 && x(2) < 0.0) ||
          (x(2) == 0.0 && x(0) < 0.0 && x(1) < 0.0))
      {
         std::vector<std::complex<double>> Eval(E.Size());
         maxwell_ess_data(x, Eval);
         for (int i = 0; i < dim; ++i)
         {
            E[i] = Eval[i].real();
         }
      }
   }
   else if (prob == waveguide)
   {
      // waveguide problem
      std::vector<std::complex<double>> Eval(E.Size());
      maxwell_ess_data(x, Eval);
      for (int i = 0; i < dim; ++i)
      {
         E[i] = Eval[i].real();
      }
      if (abs(x(0) - p.domain_bdr(0, 1)) < 1e-13)
      {
         E = 0.0;
      }
   }
}

//define bdr_data solution
void E_bdr_data_Im(const Vector &x, Vector &E)
{
   E = 0.0;
   bool in_pml = false;
   if (prob == scatter)
   {
      for (int i = 0; i < dim; ++i)
      {
         if (abs(x(i) - p.domain_bdr(i, 0)) < 1e-13 ||
             abs(x(i) - p.domain_bdr(i, 1)) < 1e-13)
         {
            in_pml = true;
            break;
         }
      }
      if (!in_pml)
      {
         std::vector<std::complex<double>> Eval(E.Size());
         maxwell_ess_data(x, Eval);
         for (int i = 0; i < dim; ++i)
         {
            E[i] = Eval[i].imag();
         }
      }
   }
   else if (prob == lshape)
   {
      if ((x(0) == 1.0 && x(1) < 1.0) ||
          (x(1) == 1.0 && x(0) < 1.0))
      {
         std::vector<std::complex<double>> Eval(E.Size());
         maxwell_ess_data(x, Eval);
         for (int i = 0; i < dim; ++i)
         {
            E[i] = Eval[i].imag();
         }
      }
   }
   else if (prob == fichera)
   {
      if ((x(0) == 0.0 && x(1) < 0.0 && x(2) < 0.0) ||
          (x(1) == 0.0 && x(0) < 0.0 && x(2) < 0.0) ||
          (x(2) == 0.0 && x(0) < 0.0 && x(1) < 0.0))
      {
         std::vector<std::complex<double>> Eval(E.Size());
         maxwell_ess_data(x, Eval);
         for (int i = 0; i < dim; ++i)
         {
            E[i] = Eval[i].imag();
         }
      }
   }
   else if (prob == waveguide)
   {
      // waveguide problem
      std::vector<std::complex<double>> Eval(E.Size());
      maxwell_ess_data(x, Eval);
      for (int i = 0; i < dim; ++i)
      {
         E[i] = Eval[i].imag();
      }
      if (abs(x(0) - p.domain_bdr(0, 1)) < 1e-13)
      {
         E = 0.0;
      }
   }
}

// PML
void pml_function(const Vector &x, std::vector<std::complex<double>> &dxs)
{
   std::complex<double> zi = std::complex<double>(0., 1.);
   std::complex<double> one = std::complex<double>(1., 0.);

   double n = 2.0;
   double c = 5.0;
   double coeff;

   // initialize to one
   for (int i = 0; i < dim; ++i)
   {
      dxs[i] = one;
   }

   // Stretch in each direction independenly
   for (int i = 0; i < dim; ++i)
   {
      for (int j = 0; j < 2; ++j)
      {
         if (x(i) >= p.comp_domain_bdr(i, 1))
         {
            coeff = n * c / omega / pow(p.length(i, 1), n);
            dxs[i] = one + zi * coeff * abs(pow(x(i) - p.comp_domain_bdr(i, 1), n - 1.0));
         }
         if (x(i) <= p.comp_domain_bdr(i, 0))
         {
            coeff = n * c / omega / pow(p.length(i, 0), n);
            dxs[i] = one + zi * coeff * abs(pow(x(i) - p.comp_domain_bdr(i, 0), n - 1.0));
         }
      }
   }
}

double pml_detJ_inv_Re(const Vector &x)
{
   std::complex<double> one = std::complex<double>(1., 0.);
   std::vector<std::complex<double>> dxs(dim);
   complex<double> det(1.0, 0.0);
   pml_function(x, dxs);
   for (int i = 0; i < dim; ++i)
   {
      det *= dxs[i];
   }

   complex<double> det_inv = one / det;
   return det_inv.real();
}
double pml_detJ_inv_Im(const Vector &x)
{
   std::complex<double> one = std::complex<double>(1., 0.);
   std::vector<std::complex<double>> dxs(dim);
   complex<double> det(1.0, 0.0);
   pml_function(x, dxs);
   for (int i = 0; i < dim; ++i)
   {
      det *= dxs[i];
   }

   complex<double> det_inv = one / det;
   return det_inv.imag();
}
void pml_detJ_JT_J_inv_Re(const Vector &x, DenseMatrix &M)
{
   std::complex<double> one = std::complex<double>(1., 0.);
   std::vector<complex<double>> diag(dim);
   std::vector<std::complex<double>> dxs(dim);
   complex<double> det(1.0, 0.0);
   pml_function(x, dxs);

   for (int i = 0; i < dim; ++i)
   {
      diag[i] = one / pow(dxs[i], 2);
      det *= dxs[i];
   }

   M.SetSize(dim);
   M = 0.0;

   for (int i = 0; i < dim; ++i)
   {
      complex<double> temp = det * diag[i];
      M(i, i) = temp.real();
   }
}
void pml_detJ_JT_J_inv_Im(const Vector &x, DenseMatrix &M)
{
   std::complex<double> one = std::complex<double>(1., 0.);
   std::vector<std::complex<double>> diag(dim);
   std::vector<std::complex<double>> dxs(dim);
   complex<double> det = 1.0;
   pml_function(x, dxs);

   for (int i = 0; i < dim; ++i)
   {
      diag[i] = one / pow(dxs[i], 2);
      det *= dxs[i];
   }

   M.SetSize(dim);
   M = 0.0;

   for (int i = 0; i < dim; ++i)
   {
      complex<double> temp = det * diag[i];
      M(i, i) = temp.imag();
   }
}
void pml_detJ_inv_JT_J_Re(const Vector &x, DenseMatrix &M)
{
   std::vector<complex<double>> diag(dim);
   std::vector<std::complex<double>> dxs(dim);
   complex<double> det(1.0, 0.0);
   pml_function(x, dxs);

   for (int i = 0; i < dim; ++i)
   {
      diag[i] = pow(dxs[i], 2);
      det *= dxs[i];
   }

   M.SetSize(dim);
   M = 0.0;
   for (int i = 0; i < dim; ++i)
   {
      complex<double> temp = diag[i] / det;
      M(i, i) = temp.real();
   }
}
void pml_detJ_inv_JT_J_Im(const Vector &x, DenseMatrix &M)
{
   std::vector<std::complex<double>> diag(dim);
   std::vector<std::complex<double>> dxs(dim);
   complex<double> det = 1.0;
   pml_function(x, dxs);

   for (int i = 0; i < dim; ++i)
   {
      diag[i] = pow(dxs[i], 2);
      det *= dxs[i];
   }

   M.SetSize(dim);
   M = 0.0;
   for (int i = 0; i < dim; ++i)
   {
      complex<double> temp = diag[i] / det;
      M(i, i) = temp.imag();
   }
}

pml::pml(Mesh *mesh_) : mesh(mesh_)
{
   //initialize to default values
   dim = mesh->Dimension();
   length.SetSize(dim, 2);
   length = 0.25;
   comp_domain_bdr.SetSize(dim, 2);
   domain_bdr.SetSize(dim, 2);
}

void pml::SetMesh(Mesh *mesh_)
{
   mesh = mesh_;
   dim = mesh->Dimension();
   length.SetSize(dim, 2);
   length = 0.25;
   comp_domain_bdr.SetSize(dim, 2);
   domain_bdr.SetSize(dim, 2);
}

void pml::SetPmlLength(Array2D<double> &length_)
{
   // check dimensions
   MFEM_VERIFY(length_.NumRows() == dim, " Pml length NumRows missmatch ");
   MFEM_VERIFY(length_.NumCols() == 2, " Pml length NumCols missmatch ");
   length = length_;
}

void pml::compute_data()
{
   // initialize with any vertex
   for (int i = 0; i < dim; i++)
   {
      domain_bdr(i, 0) = mesh->GetVertex(0)[i];
      domain_bdr(i, 1) = mesh->GetVertex(0)[i];
   }
   // loop through boundary vertices
   for (int i = 0; i < mesh->GetNBE(); i++)
   {
      Array<int> bdr_vertices;
      mesh->GetBdrElementVertices(i, bdr_vertices);
      // loop through vertices
      for (int j = 0; j < bdr_vertices.Size(); j++)
      {
         for (int k = 0; k < dim; k++)
         {
            domain_bdr(k, 0) = min(domain_bdr(k, 0), mesh->GetVertex(bdr_vertices[j])[k]);
            domain_bdr(k, 1) = max(domain_bdr(k, 1), mesh->GetVertex(bdr_vertices[j])[k]);
         }
      }
   }
   for (int i = 0; i < dim; i++)
   {
      comp_domain_bdr(i, 0) = domain_bdr(i, 0) + length(i, 0);
      comp_domain_bdr(i, 1) = domain_bdr(i, 1) - length(i, 1);
   }
}

void pml::get_pml_elem_list(ParMesh *pmesh, Array<int> &elem_pml)
{
   int nrelem = pmesh->GetNE();
   // initialize list with 1
   elem_pml.SetSize(nrelem);
   elem_pml = 1;
   // loop through the elements and identify which of them are in the pml
   for (int i = 0; i < nrelem; ++i)
   {
      Element *el = pmesh->GetElement(i);
      Array<int> vertices;
      el->GetVertices(vertices);
      int nrvert = vertices.Size();
      // Check if any vertex is in the pml
      for (int iv = 0; iv < nrvert; ++iv)
      {
         int vert_idx = vertices[iv];
         double *coords = pmesh->GetVertex(vert_idx);
         for (int comp = 0; comp < dim; ++comp)
         {
            if (coords[comp] > comp_domain_bdr(comp, 1) ||
                coords[comp] < comp_domain_bdr(comp, 0))
            {
               elem_pml[i] = 0;
            }
         }
      }
   }
}