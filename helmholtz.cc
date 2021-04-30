/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2021 by Wolfgang Bangerth and SAATI Co.
 *
 * ---------------------------------------------------------------------
 */


#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/thread_management.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/timer.h>

#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparse_direct.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/reference_cell.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>

#include <deal.II/fe/fe_interface_values.h>
#include <deal.II/meshworker/mesh_loop.h>

#include <fstream>
#include <future>
#include <thread>
#include <iostream>
#include <sstream>
#include <cmath>
#include <cstdio>
#include <complex>
#include <memory>

std::string instance_folder;
std::ofstream logger;

namespace TransmissionProblem
{
  using namespace dealii;

  using ScalarType = std::complex<double>;

  // The following namespace defines material parameters. We use SI units.
  namespace MaterialParameters
  {
    std::complex<double> wave_speed;

    std::vector<double> frequencies;
  }

  std::string mesh_file_name;
  
  unsigned int fe_degree = 2;
  int n_mesh_refinement_steps = 5;

  unsigned int n_threads = 0;

  void
  declare_parameters (ParameterHandler &prm)
  {
    prm.declare_entry ("Mesh file name", "./mesh.msh",
                       Patterns::FileName(),
                       "The name of the file from which to read the mesh.");
    prm.declare_entry ("Wave speed", "340",
                       Patterns::Double(0),
                       "The wave speed in the medium in question. Units: [m/s].");
    prm.declare_entry ("Wave speed loss tangent", "20",
                       Patterns::Double(0,90),
                       "The angle used to make the wave speed complex-valued. "
                       "Units: [degrees].");

    prm.declare_entry ("Frequencies", "linear_spacing(100,10000,100)",
                       Patterns::Anything(),
                       "A description of the frequencies to compute. See "
                       "the readme.md file for a description of the format "
                       "for this entry.");

    prm.declare_entry ("Number of mesh refinement steps", "5",
                       Patterns::Integer(-100,10),
                       "The number of global mesh refinement steps applied "
                       "to the coarse mesh if positive or zero. If negative, "
                       "then it denotes the number of mesh points per wave length "
                       "as described in readme.md.");
    prm.declare_entry ("Finite element polynomial degree", "2",
                       Patterns::Integer(1,5),
                       "The polynomial degree to be used for the finite element.");

    prm.declare_entry ("Number of threads", "0",
                       Patterns::Integer(0),
                       "The number of threads this program may use at the same time. "
                       "Threads are used to compute the frequency response for "
                       "different frequencies at the same time since these are "
                       "independent computations. A value of zero means that the "
                       "program may use as many threads as it pleases, whereas a "
                       "positive number limits how many threads (and consequently "
                       "CPU cores) the program will use.");
  }


  void
  read_parameters (ParameterHandler &prm)
  {
    // First read parameter values from the input file 'helmholtz.prm'
    prm.parse_input (instance_folder + "/helmholtz.prm");
    

    
    using namespace MaterialParameters;

    // Read the (real-valued) wave speed and its loss tangent. Then
    // make the wave speed complex-valued.
    const double c              = prm.get_double ("Wave speed");
    const double c_loss_tangent = prm.get_double ("Wave speed loss tangent");

    wave_speed = c * std::exp(std::complex<double>(0,2*numbers::PI*c_loss_tangent/360));
    

    mesh_file_name = prm.get ("Mesh file name");

    // Read and parse the entry that determines which frequencies to compute.
    // Recall that the format is one of the following:
    // - linear_spacing(min,max,n_steps)
    // - exp_spacing(min,max,n_steps)
    // - list(...)
    const std::string frequency_descriptor = prm.get ("Frequencies");
    if (frequency_descriptor.find ("linear_spacing") == 0)
      {
        // Get the rest of the string, and eat any space at the start and end
        const std::string parenthesized_expr
          = Utilities::trim (frequency_descriptor.substr
                             (std::string("linear_spacing").size(),
                              std::string::npos));
        AssertThrow (parenthesized_expr.size() >= 2
                     &&
                     parenthesized_expr.front() == '('
                     &&
                     parenthesized_expr.back() == ')',
                     ExcMessage ("Wrong format for 'linear_spacing'."));

        // Then get the interior part, again trim spaces, and split at
        // commas
        const std::vector<std::string> min_max_steps
          = Utilities::split_string_list
          (Utilities::trim (parenthesized_expr.substr
                            (1,
                             parenthesized_expr.size() - 2)),
           ',');
        AssertThrow (min_max_steps.size() == 3,
                     ExcMessage ("Wrong format for 'linear_spacing'."));
                            
        const double min_omega = Utilities::string_to_double(min_max_steps[0])
                                 * 2 * numbers::PI;
        const double max_omega = Utilities::string_to_double(min_max_steps[1])
                                 * 2 * numbers::PI;
        const unsigned int n_frequencies = Utilities::string_to_int(min_max_steps[2]);

        const double delta_omega = (max_omega - min_omega)
                                   / (n_frequencies-1)
                                   * (1.-1e-12);
        for (double omega = min_omega;
             omega <= max_omega;
             omega += delta_omega)
          MaterialParameters::frequencies.push_back (omega);
      }
    else if (frequency_descriptor.find ("exp_spacing") == 0)
      {
        // Get the rest of the string, and eat any space at the start and end
        const std::string parenthesized_expr
          = Utilities::trim (frequency_descriptor.substr
                             (std::string("exp_spacing").size(),
                              std::string::npos));
        AssertThrow (parenthesized_expr.size() >= 2
                     &&
                     parenthesized_expr.front() == '('
                     &&
                     parenthesized_expr.back() == ')',
                     ExcMessage ("Wrong format for 'exp_spacing'."));

        // Then get the interior part, again trim spaces, and split at
        // commas
        const std::vector<std::string> min_max_steps
          = Utilities::split_string_list
          (Utilities::trim (parenthesized_expr.substr
                            (1,
                             parenthesized_expr.size() - 2)),
           ',');
        AssertThrow (min_max_steps.size() == 3,
                     ExcMessage ("Wrong format for 'exp_spacing'."));
                            
        const double log_min_omega = std::log(Utilities::string_to_double(min_max_steps[0])
                                              * 2 * numbers::PI);
        const double log_max_omega = std::log(Utilities::string_to_double(min_max_steps[1])
                                              * 2 * numbers::PI);
        const unsigned int n_frequencies = Utilities::string_to_int(min_max_steps[2]);

        const double delta_log_omega = (log_max_omega - log_min_omega)
                                       / (n_frequencies - 1)
                                       * (1.-1e-12);
        for (double log_omega = log_min_omega;
             log_omega <= log_max_omega;
             log_omega += delta_log_omega)
          MaterialParameters::frequencies.push_back (std::exp(log_omega));
      }
    else if (frequency_descriptor.find ("list") == 0)
      {
        // Get the rest of the string, and eat any space at the start and end
        const std::string parenthesized_expr
          = Utilities::trim (frequency_descriptor.substr
                             (std::string("list").size(),
                              std::string::npos));
        AssertThrow (parenthesized_expr.size() >= 2
                     &&
                     parenthesized_expr.front() == '('
                     &&
                     parenthesized_expr.back() == ')',
                     ExcMessage ("Wrong format for 'list' frequency spacing."));

        // Then get the interior part, again trim spaces, and split at
        // commas
        MaterialParameters::frequencies =
          Utilities::string_to_double
          (Utilities::split_string_list
           (Utilities::trim (parenthesized_expr.substr
                             (1,
                              parenthesized_expr.size() - 2)),
            ','));
        AssertThrow (MaterialParameters::frequencies.size() >= 1,
                     ExcMessage ("Wrong format for 'list' frequency spacing."));

        // Because MaterialParameters::frequencies stores angular
        // frequencies, we need to multiply by 2*pi
        for (auto &f : MaterialParameters::frequencies)
          f *= 2 * numbers::PI;
      }
    else
      AssertThrow (false,
                   ExcMessage ("The format for the description of the frequencies to "
                               "be solved for, namely <"
                               + frequency_descriptor + ">, did not match any of "
                               "the recognized formats."));    

    fe_degree               = prm.get_integer ("Finite element polynomial degree");
    n_mesh_refinement_steps = prm.get_integer ("Number of mesh refinement steps");

    Assert(fe_degree >= 2,
           ExcMessage("The C0IP formulation for the helmholtz problem "
                      "only works if one uses elements of polynomial "
                      "degree at least 2."));

    n_threads = prm.get_integer ("Number of threads");
  }
  

  

  
  // A data structure that is used to collect the results of the computations
  // for one frequency. The main class fills this for a given frequency
  // in various places of its member functions, and at the end puts it into
  // a global map.
  struct OutputData
  {
    FullMatrix<ScalarType> P, U;

    std::vector<std::string> visualization_file_names;
  };
  
    
  
  // A variable that will collect the data (value) for all frequencies
  // omega (key). Since we will access it from different threads, we also
  // need a mutex to guard access to it.
  std::map<double,OutputData> results;
  std::mutex results_mutex;

  TimerOutput timer_output = TimerOutput (logger, TimerOutput::summary,
                                          TimerOutput::wall_times);
  

  // Check whether an external program has left a signal that
  // indicates that the current program run should terminate without
  // computing any further frequency responses. This is done by
  // placing the word "STOP" into the file "termination_signal" in the
  // current directory.
  //
  // Once detected, we delete the file again and terminate the
  // program.
  bool check_for_termination_signal()
  {
    static bool termination_requested = false;

    static std::mutex mutex;
    std::lock_guard<std::mutex> lock_guard (mutex);

    if (termination_requested == true)
      return true;
    
    // Try and see whether we can open the file at all. If we can't,
    // then no termination signal has been sent. If so, return 'true',
    // but before that set a flag that ensures we don't have to do the
    // expensive test with the file in any further calls. (We'll try
    // to abort the program below, but this may block for a bit
    // because we need to wait for the lock that guards access to the
    // output file.)
    std::ifstream in(instance_folder + "/termination_signal");
    if (!in)
      {
        termination_requested = false;
        return false;
      }

    // OK, the file exists, but does it contain the right content?
    std::string line;
    std::getline(in, line);
    if (line == "STOP")
      {
        termination_requested = true;

        // Close the file handle and remove the file.
        in.close();
        std::remove ((instance_folder + "/termination_signal").c_str());

        // Now wait for the lock that guards access to the output file
        // and if we have it, we know that nothing else is writing to
        // the file at the moment and we can safely abort the program.
        std::lock_guard<std::mutex> results_lock(results_mutex);
        logger << "INFO *** Terminating program upon request." << std::endl;
        std::exit (1);
        
        return true;
      }

    // The file exists, but it has the wrong content (or no content so
    // far). This means no termination. In the best of all cases, we
    // will have caught the driver program having created but not
    // written to the file. The next time we check, we might find the
    // file in the correct state.
    return false;
  }



  // @sect3{The main class}
  //
  // The following is the principal class of this tutorial program. It has
  // the structure of many of the other tutorial programs and there should
  // really be nothing particularly surprising about its contents or
  // the constructor that follows it.
  template <int dim>
  class HelmholtzProblem
  {
  public:
    HelmholtzProblem(const double omega);

    void run();

  private:
    void make_grid();
    void setup_system();
    void assemble_system(const unsigned int current_source_port);
    void solve();
    void postprocess(const unsigned int current_source_port);
    void output_results(const unsigned int current_source_port);

    // The frequency that this instance of the class is supposed to solve for.
    const double                  omega;

    Triangulation<dim>              triangulation;
    std::vector<types::boundary_id> port_boundary_ids;
    std::vector<double>             port_areas;
    
    std::unique_ptr<Mapping<dim>> mapping;

    FE_SimplexP<dim>              fe;
    DoFHandler<dim>               dof_handler;

    SparsityPattern               sparsity_pattern;
    SparseMatrix<ScalarType>      system_matrix;

    Vector<ScalarType>            solution;
    Vector<ScalarType>            system_rhs;

    OutputData                    output_data;
  };



  template <int dim>
  HelmholtzProblem<dim>::HelmholtzProblem(const double omega)
    : omega (omega)
      , mapping(ReferenceCells::Tetrahedron.get_default_mapping<dim,dim>(1))
    , fe(TransmissionProblem::fe_degree)
    , dof_handler(triangulation)
  {}



  // Next up are the functions that create the initial mesh (a once refined
  // unit square) and set up the constraints, vectors, and matrices on
  // each mesh. Again, both of these are essentially unchanged from many
  // previous tutorial programs.
  template <int dim>
  void HelmholtzProblem<dim>::make_grid()
  {
    std::unique_ptr<TimerOutput::Scope> timer_section = (n_threads==1 ? std::make_unique<TimerOutput::Scope>(timer_output, "Make grid") : nullptr);
    
    GridIn<dim> grid_in;
    grid_in.attach_triangulation (triangulation);
    std::ifstream input (instance_folder + "/" + mesh_file_name);
    grid_in.read_msh (input);

    std::cout << "The mesh has " << triangulation.n_active_cells() << " cells" << std::endl;
    
    // Now implement the heuristic for mesh refinement described in
    // readme.md: If positive, just do a number of global refinement
    // steps. If negative, interpret it as the number of mesh points
    // per wave length.
    if (n_mesh_refinement_steps >= 0)
      triangulation.refine_global(n_mesh_refinement_steps);
    else
      {
        Assert (false, ExcMessage("Need to implement a scheme that is based on the wavelength"));
        
        const int N = -n_mesh_refinement_steps;
        
        const double lambda  = 1000;

        const double diameter = GridTools::diameter(triangulation);
        const double delta_x = std::min(lambda, diameter) / N * fe_degree;

        while (GridTools::maximal_cell_diameter(triangulation)
               >= delta_x)
          triangulation.refine_global();
      }

    // Figure out what boundary ids we have that describe ports. We
    // take these as all of those boundary ids that are non-zero
    port_boundary_ids = {0,1,2}, // TODO: triangulation.get_boundary_ids();
    port_boundary_ids.erase (std::find(port_boundary_ids.begin(),
                                       port_boundary_ids.end(),
                                       types::boundary_id(0)));

    // Now also correctly size the matrices we compute for each
    // frequency
    output_data.P.reinit (port_boundary_ids.size(),
                          port_boundary_ids.size());
    output_data.U.reinit (port_boundary_ids.size(),
                          port_boundary_ids.size());

    // As a final step, compute the areas of the various ports so we
    // can later normalize when computing average pressures and
    // velocities:
    port_areas.resize (port_boundary_ids.size(), 0.);
    const QGauss<dim-1>  quadrature_formula(fe.degree + 2);
    const unsigned int n_q_points = quadrature_formula.size();
    FEFaceValues<dim> fe_face_values(*mapping,
                                     fe,
                                     quadrature_formula,
                                     update_JxW_values);
    for (const auto cell : dof_handler.active_cell_iterators())
      for (const auto face : cell->face_iterators())
        if (face->at_boundary())
          if (std::find(port_boundary_ids.begin(),
                        port_boundary_ids.end(),
                        face->boundary_id()) !=
              port_boundary_ids.end()) // a port
            {
              const unsigned int this_port = (std::find(port_boundary_ids.begin(),
                                                        port_boundary_ids.end(),
                                                        face->boundary_id())
                                              - port_boundary_ids.begin());
              
              fe_face_values.reinit(cell, face);
              for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
                port_areas[this_port] += fe_face_values.JxW(q_point);
            }
  }



  template <int dim>
  void HelmholtzProblem<dim>::setup_system()
  {
    std::unique_ptr<TimerOutput::Scope> timer_section = (n_threads==1 ? std::make_unique<TimerOutput::Scope>(timer_output, "Set up system") : nullptr);

    dof_handler.distribute_dofs(fe);

    std::cout << "The mesh has " << dof_handler.n_dofs() << " unknowns" << std::endl;
    
    DynamicSparsityPattern c_sparsity(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern (dof_handler, c_sparsity);
    c_sparsity.compress();
    
        
    sparsity_pattern.copy_from(c_sparsity);
    system_matrix.reinit(sparsity_pattern);

    solution.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());
  }



  // @sect{Assembling the linear system}
  //
  // The following pieces of code are more interesting. They all relate to the
  // assembly of the linear system. While assemling the cell-interior terms
  // is not of great difficulty -- that works in essence like the assembly
  // of the corresponding terms of the Laplace equation, and you have seen
  // how this works in step-4 or step-6, for example -- the difficulty
  // is with the penalty terms in the formulation. These require the evaluation
  // of gradients of shape functions at interfaces of cells. At the least,
  // one would therefore need to use two FEFaceValues objects, but if one of the
  // two sides is adaptively refined, then one actually needs an FEFaceValues
  // and one FESubfaceValues objects; one also needs to keep track which
  // shape functions live where, and finally we need to ensure that every
  // face is visited only once. All of this is a substantial overhead to the
  // logic we really want to implement (namely the penalty terms in the
  // bilinear form). As a consequence, we will make use of the
  // FEInterfaceValues class -- a helper class in deal.II that allows us
  // to abstract away the two FEFaceValues or FESubfaceValues objects and
  // directly access what we really care about: jumps, averages, etc.
  //
  // But this doesn't yet solve our problem of having to keep track of
  // which faces we have already visited when we loop over all cells and
  // all of their faces. To make this process simpler, we use the
  // MeshWorker::mesh_loop() function that provides a simple interface
  // for this task: Based on the ideas outlined in the WorkStream
  // namespace documentation, MeshWorker::mesh_loop() requires three
  // functions that do work on cells, interior faces, and boundary
  // faces; these functions work on scratch objects for intermediate
  // results, and then copy the result of their computations into
  // copy data objects from where a copier function copies them into
  // the global matrix and right hand side objects.
  //
  // The following structures then provide the scratch and copy objects
  // that are necessary for this approach. You may look up the WorkStream
  // namespace as well as the
  // @ref threads "Parallel computing with multiple processors"
  // module for more information on how they typically work.
  template <int dim>
  struct ScratchData
  {
    ScratchData(const Mapping<dim> &      mapping,
                const FiniteElement<dim> &fe,
                const unsigned int        quadrature_degree,
                const UpdateFlags         update_flags = update_values |
                                                 update_gradients |
                                                 update_quadrature_points |
                                                 update_JxW_values,
                const UpdateFlags face_update_flags =
                  update_values | update_gradients | update_quadrature_points |
                  update_JxW_values | update_normal_vectors)
      : fe_values(mapping, fe, QGauss<dim>(quadrature_degree), update_flags)
      , fe_face_values(mapping,
                       fe,
                       QGauss<dim - 1>(quadrature_degree),
                       face_update_flags)
    {}


    ScratchData(const ScratchData<dim> &scratch_data)
      : fe_values(scratch_data.fe_values.get_mapping(),
                  scratch_data.fe_values.get_fe(),
                  scratch_data.fe_values.get_quadrature(),
                  scratch_data.fe_values.get_update_flags())
      , fe_face_values(scratch_data.fe_values.get_mapping(),
                       scratch_data.fe_values.get_fe(),
                       scratch_data.fe_face_values.get_quadrature(),
                       scratch_data.fe_face_values.get_update_flags())
    {}

    FEValues<dim>     fe_values;
    FEFaceValues<dim> fe_face_values;
  };



  struct CopyData
  {
    CopyData(const unsigned int dofs_per_cell)
      : cell_matrix(dofs_per_cell, dofs_per_cell)
      , cell_rhs(dofs_per_cell)
      , local_dof_indices(dofs_per_cell)
    {}


    CopyData(const CopyData &) = default;


    struct FaceData
    {
      FullMatrix<ScalarType>               cell_matrix;
      std::vector<types::global_dof_index> joint_dof_indices;
    };

    FullMatrix<ScalarType>               cell_matrix;
    Vector<ScalarType>                   cell_rhs;

    std::vector<types::global_dof_index> local_dof_indices;
    std::vector<FaceData>                face_data;
  };



  template <int dim>
  void HelmholtzProblem<dim>::assemble_system(const unsigned int current_source_port)
  {
    std::unique_ptr<TimerOutput::Scope> timer_section = (n_threads==1 ? std::make_unique<TimerOutput::Scope>(timer_output, "Assemble linear system") : nullptr);

    using Iterator = typename DoFHandler<dim>::active_cell_iterator;

    auto cell_worker = [&](const Iterator &  cell,
                           ScratchData<dim> &scratch_data,
                           CopyData &        copy_data) {
      std::unique_ptr<TimerOutput::Scope> timer_section = (n_threads==1 ? std::make_unique<TimerOutput::Scope>(timer_output, "Assemble linear system - cell") : nullptr);
      
      copy_data.cell_matrix = 0;
      copy_data.cell_rhs    = 0;

      scratch_data.fe_values.reinit(cell);
      cell->get_dof_indices(copy_data.local_dof_indices);

      const FEValues<dim> &fe_values = scratch_data.fe_values;

      const unsigned int dofs_per_cell =
        scratch_data.fe_values.get_fe().dofs_per_cell;

      for (unsigned int qpoint = 0; qpoint < fe_values.n_quadrature_points;
           ++qpoint)
        {
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              const Tensor<1,dim> grad_i    = fe_values.shape_grad(i, qpoint);
              const double        value_i   = fe_values.shape_value(i, qpoint);
              
              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                  const Tensor<1,dim> grad_j    = fe_values.shape_grad(j, qpoint);
                  const double        value_j   = fe_values.shape_value(j, qpoint);

                  copy_data.cell_matrix(i, j) +=
                    (
                     +
                     MaterialParameters::wave_speed *
                     MaterialParameters::wave_speed *
                     grad_i *
                     grad_j
                     -
                     omega *
                     omega *
                     value_i *
                     value_j
                     )
                    * fe_values.JxW(qpoint);                  // dx
                }

              copy_data.cell_rhs(i) +=
                fe_values.shape_value(i, qpoint) * // phi_i(x)
                0 * // f(x)=0
                fe_values.JxW(qpoint);                  // dx
            }
        }
    };
    


    // Part 4 was a small function that copies the data produced by the
    // cell, interior, and boundary face assemblers above into the
    // global matrix and right hand side vector. There really is not
    // very much to do here: We distribute the cell matrix and right
    // hand side contributions as we have done in almost all of the
    // other tutorial programs using the constraints objects. We then
    // also have to do the same for the face matrix contributions
    // that have gained content for the faces (interior and boundary)
    // and that the `face_worker` and `boundary_worker` have added
    // to the `copy_data.face_data` array.
    auto copier = [&](const CopyData &copy_data) {
      std::unique_ptr<TimerOutput::Scope> timer_section = (n_threads==1 ? std::make_unique<TimerOutput::Scope>(timer_output, "Assemble linear system - copy") : nullptr);
      
      for (unsigned int i=0; i<copy_data.cell_matrix.m(); ++i)
        for (unsigned int j=0; j<copy_data.cell_matrix.m(); ++j)
          system_matrix.add(copy_data.local_dof_indices[i],
                            copy_data.local_dof_indices[j],
                            copy_data.cell_matrix(i,j));
      for (unsigned int i=0; i<copy_data.cell_rhs.size(); ++i)
        system_rhs(copy_data.local_dof_indices[i])
          += copy_data.cell_rhs[i];

      for (auto &cdf : copy_data.face_data)
        {
          for (unsigned int i=0; i<cdf.cell_matrix.m(); ++i)
            for (unsigned int j=0; j<cdf.cell_matrix.m(); ++j)
              system_matrix.add(cdf.joint_dof_indices[i],
                                cdf.joint_dof_indices[j],
                                cdf.cell_matrix(i,j));
        }
    };


    // Having set all of this up, what remains is to just create a scratch
    // and copy data object and call the MeshWorker::mesh_loop() function
    // that then goes over all cells and faces, calls the respective workers
    // on them, and then the copier function that puts things into the
    // global matrix and right hand side. As an additional benefit,
    // MeshWorker::mesh_loop() does all of this in parallel, using
    // as many processor cores as your machine happens to have.
    const unsigned int n_gauss_points = dof_handler.get_fe().degree + 1;
    ScratchData<dim>   scratch_data(*mapping,
                                    fe,
                                    n_gauss_points,
                                    update_values | update_gradients |
                                    update_JxW_values,
                                    update_default);
    CopyData           copy_data(dof_handler.get_fe().dofs_per_cell);
    MeshWorker::mesh_loop(dof_handler.begin_active(),
                          dof_handler.end(),
                          cell_worker,
                          copier,
                          scratch_data,
                          copy_data,
                          MeshWorker::assemble_own_cells,
                          /* face_worker= */ {},
                          /* face_worker= */ {});

    std::map<types::global_dof_index,ScalarType> boundary_values;
    VectorTools::interpolate_boundary_values(dof_handler,
                                             port_boundary_ids[current_source_port],
                                             Functions::ConstantFunction<dim,ScalarType>(1),
                                             boundary_values);

    MatrixTools::apply_boundary_values (boundary_values,
                                        system_matrix,
                                        solution,
                                        system_rhs);
  }



  // @sect{Solving the linear system and postprocessing}
  //
  // The show is essentially over at this point: The remaining functions are
  // not overly interesting or novel. The first one simply uses a direct
  // solver to solve the linear system (see also step-29):
  template <int dim>
  void HelmholtzProblem<dim>::solve()
  {
    std::unique_ptr<TimerOutput::Scope> timer_section = (n_threads==1 ? std::make_unique<TimerOutput::Scope>(timer_output, "Solve linear system") : nullptr);

    solution = system_rhs;
    
    SparseDirectUMFPACK direct_solver;
    direct_solver.solve(system_matrix, solution);
  }



  // The next function postprocesses the solution. In the current context,
  // this implies computing the integral over the magnitude of the solution.
  // It will be small in general, except in the vicinity of eigenvalues.
  template <int dim>
  void HelmholtzProblem<dim>::postprocess(const unsigned int current_source_port)
  {
    std::unique_ptr<TimerOutput::Scope> timer_section = (n_threads==1 ? std::make_unique<TimerOutput::Scope>(timer_output, "Postprocess") : nullptr);

    // Compute the integral of the absolute value of the solution.
    const QGauss<dim-1>  quadrature_formula(fe.degree + 2);
    const unsigned int n_q_points = quadrature_formula.size();
    FEFaceValues<dim> fe_face_values(*mapping,
                                     fe,
                                     quadrature_formula,
                                     update_values
                                     | update_gradients
                                     | update_quadrature_points
                                     | update_normal_vectors
                                     | update_JxW_values);

    std::vector<ScalarType> solution_values(n_q_points);
    std::vector<Tensor<1,dim,ScalarType>> solution_gradients(n_q_points);
    for (const auto cell : dof_handler.active_cell_iterators())
      for (const auto face : cell->face_iterators())
        if (face->at_boundary())
          if (std::find(port_boundary_ids.begin(),
                        port_boundary_ids.end(),
                        face->boundary_id()) !=
              port_boundary_ids.end()) // a port
            {
              const unsigned int this_port = (std::find(port_boundary_ids.begin(),
                                                        port_boundary_ids.end(),
                                                        face->boundary_id())
                                              - port_boundary_ids.begin());
              
              fe_face_values.reinit(cell, face);
              fe_face_values.get_function_values   (solution, solution_values);
              fe_face_values.get_function_gradients(solution, solution_gradients);
              
            
              for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
                {
                  // Compute the integral over the pressure on each
                  // port (including the source port, where we should
                  // eventually get an average pressure of one):
                  output_data.P(current_source_port, this_port)
                    +=  (solution_values[q_point] *
                         fe_face_values.JxW(q_point));

                  // Then also compute the integral over the
                  // velocity. Since the pressure has units
                  // Pa=kg/m/s^2, we get the velocity by
                  //   U=...
                  // with units
                  //     
                  const auto velocity = omega * solution_gradients[q_point];
                  
                  output_data.U(current_source_port, this_port)
                    +=  (velocity *
                         fe_face_values.normal_vector(q_point) *
                         fe_face_values.JxW(q_point));
                }
            }

    // Finally, normalize this column of the matrices by the area so
    // we get averages:
    for (unsigned int i=0; i<port_areas.size(); ++i)
      {
        output_data.P(current_source_port, i) /= port_areas[i];
        output_data.U(current_source_port, i) /= port_areas[i];
      }
  }



  // Equally uninteresting is the function that generates graphical output.
  // It looks exactly like the one in step-6, for example.
  template <int dim>
  void
  HelmholtzProblem<dim>::output_results(const unsigned int current_source_port)
  {
    std::unique_ptr<TimerOutput::Scope> timer_section = (n_threads==1 ? std::make_unique<TimerOutput::Scope>(timer_output, "Creating visual output") : nullptr);

    DataOut<dim> data_out;

    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution, "solution");
    data_out.build_patches(fe.degree);

    std::string file_name = instance_folder + "/visualization/solution-" +
                            std::to_string(static_cast<unsigned int>(omega/2/numbers::PI)) +
                            "." +
                            std::to_string(current_source_port) +
                            ".vtu";
    std::ofstream output_vtu(file_name);
    AssertThrow (output_vtu,
                 ExcMessage ("The file <" + file_name +
                             "> can not be written to when trying to write "
                             "visualization data."));
    data_out.write_vtu(output_vtu);

    output_data.visualization_file_names.emplace_back (file_name);
  }



  // The same is true for the `run()` function: Just like in previous
  // programs.
  template <int dim>
  void HelmholtzProblem<dim>::run()
  {
    check_for_termination_signal();

    make_grid();

    setup_system();

    for (unsigned int current_source_port=0;
         current_source_port<port_boundary_ids.size();
         ++current_source_port)
      {
        check_for_termination_signal();

        std::cout << "omega=" << omega
                  << ", source port boundary id=" << port_boundary_ids[current_source_port]
                  << std::endl;    
        assemble_system(current_source_port);

        check_for_termination_signal();

        solve();

        check_for_termination_signal();

        // Do postprocessing:
        output_results(current_source_port);
        postprocess(current_source_port);
      }
    
    // Finally, put the result into the output variable that we can
    // read from main(). Make sure that access to the variable is
    // properly guarded across threads.
    std::lock_guard<std::mutex> guard (results_mutex);
    results[omega] = output_data;
  }



  void solve_one_frequency (const double omega)
  {
    // The main() function has created tasks for all frequencies
    // provided by the caller, but there is the possibility that a
    // higher instance has decided that the program needs to stop
    // doing what it's doing. Check here, as this is the first
    // non-trivial place one ends up when a task executes, whether we
    // are supposed to actually do anything, or should instead stop
    // working on the frequency this task corresponds to.
    if (check_for_termination_signal() == true)
      {
        logger << "INFO Aborting work on omega = " << omega << std::endl;
        return;
      }

    try
      {
        HelmholtzProblem<3> helmholtz_problem(omega);
        helmholtz_problem.run();
      }
    catch (const std::exception &exc)
      {
        logger << "ERROR Exception while computing for frequency "
                  << omega/2/numbers::PI << ":\n"
                  << exc.what() << std::endl;
        throw;
      }


    // We have just finished another frequency. The 'run()' function
    // just called will have put its results into a shared
    // std::map. Re-create the output file based on what's now in this
    // std::map so that the current state of the computations is kept
    // up-to-date in real-time and can be looked up in the output
    // file.
    //
    // Make sure that we wait for all threads to release access to the
    // variable. The lock also makes sure that only one thread at a
    // time accesses the output file. That said, instead of writing
    // directly into the file, we first write into a buffer and then
    // dump the buffer in its entirety into the output file. That's
    // because the calling process (ARES) might want to monitor what's
    // already been computed and check in on this file
    // periodically. We would like this to happen with the file in its
    // final form, not partly written.
    std::lock_guard<std::mutex> guard (results_mutex);

    // First output how many frequencies have already been computed:
    std::ostringstream buffer;
    for (auto result : results)
      {
        const unsigned int field_width = 24;
        
        const auto omega = result.first;
        buffer << "Results for frequency f="
               << omega/2/numbers::PI << ":\n"
               << "==============================\n\n";

        buffer << "P = [\n";
        for (unsigned int i=0; i<result.second.P.m(); ++i)
          {
            buffer << "      [";
            for (unsigned int j=0; j<result.second.P.n(); ++j)
              buffer << std::setw(field_width) << result.second.P(i,j) << ' ';
            buffer << "]\n";
          }
        buffer << "]\n";
        
        buffer << "\n\nU = [\n";
        for (unsigned int i=0; i<result.second.U.m(); ++i)
          {
            buffer << "      [";
            for (unsigned int j=0; j<result.second.P.n(); ++j)
              buffer << std::setw(field_width) << result.second.U(i,j) << ' ';
            buffer << "]\n";
          }
        buffer << "]\n";
        buffer << "\n\n\n" << std::flush;
      }
    
    std::ofstream frequency_response (instance_folder + "/frequency_response.txt");
    frequency_response << buffer.str();
  }
} // namespace TransmissionProblem



// @sect3{The main() function}
//
// Finally for the `main()` function. There is, again, not very much to see
// here: It looks like the ones in previous tutorial programs. There
// is a variable that allows selecting the polynomial degree of the element
// we want to use for solving the equation. Because the C0IP formulation
// we use requires the element degree to be at least two, we check with
// an assertion that whatever one sets for the polynomial degree actually
// makes sense.
int main(int argc, char *argv[])
{
  if (argc >= 2)
    {
      instance_folder = std::string(argv[1]);
    }
  else
    {
      instance_folder = std::string(".");
    }
  
  logger = std::ofstream (instance_folder + "/output.log");
  logger << "INFO Program started with argument '" << instance_folder << "'" << std::endl;

  try
    {
      using namespace dealii;
      using namespace TransmissionProblem;

      // Set the limit for internal thread creation to one -- i.e.,
      // deal.II will not create any threads itself for things such as
      // matrix-vector products. The logic below may also create tasks
      // that run in parallel, and because each of these tasks doesn't
      // use parallelism itself, the overall parallelism is determined
      // by how many tasks we create explicitly below.
      MultithreadInfo::set_thread_limit (1);
      

      // Remove any previous output file so that nobody gets confused
      // if the program were to be aborted before we write into it the
      // first time.
      std::remove ((instance_folder + "/frequency_response.txt").c_str());

      // Get the global set of parameters from an input file
      {
        ParameterHandler prm;
        declare_parameters(prm);
        read_parameters(prm);
      }

      // Finally start the computations. If we are allowed to use as
      // many threads as we want, or if we are allowed to use as many
      // or more threads as there are frequencies, then we can just
      // schedule all of them:
      if ((n_threads == 0)
          ||
          (n_threads >= MaterialParameters::frequencies.size()))
        {
          std::vector<std::future<void>> tasks;
          for (const double omega : MaterialParameters::frequencies)
            tasks.emplace_back (std::async (std::launch::async,
                                            [=]() { solve_one_frequency (omega); }));
      
          logger << "INFO Number of frequencies scheduled: "
                    << tasks.size() << std::endl;

          // Now wait for it all:
          for (const auto &task : tasks)
            task.wait();
        }
      else
        // We are limited on the number of threads. The way we deal
        // with this is that we start a few tasks right away (as many
        // as we are allowed to) and keep a list of frequencies that
        // still need to be finished. Then, each task that finishes
        // creates a continuation just before it terminates.
        {
          std::vector<double>
            leftover_frequencies (MaterialParameters::frequencies.begin()+n_threads,
                                  MaterialParameters::frequencies.end());
          std::mutex mutex;

          // Here is the task we have to do for each of the initial
          // frequencies: Do one frequency. Then see whether there are
          // any frequencies left and if so, queue up a task for the
          // next available frequency. Accessing both
          // `leftover_frequencies` and `tasks` obviously has to
          // happen under a lock.
          //
          // Note how this lambda function calls itself.
          std::function<void (double)> do_one_frequency
            = [&] (const double omega) {
                solve_one_frequency (omega);

                double next_omega = -1e20;
                {
                  std::lock_guard<std::mutex> lock(mutex);
                  if (leftover_frequencies.size() == 0)
                    return;
                
                  next_omega = leftover_frequencies.front();
                  leftover_frequencies.erase (leftover_frequencies.begin());
                }

                // The lock has been released, we can just do the next
                // frequency, simply re-using the current thread. (The
                // load balancing will happen because each thread that
                // was initially started keeps working until there is
                // no more work to be found on the
                // `leftover_frequency` stack. In other words, we
                // don't up-front schedule which task will do what,
                // but in essence implement a work-stealing strategy.)
                do_one_frequency (next_omega);
          };

          // Now start the initial tasks.
          logger << "INFO Using processing with limited number of "
                    << n_threads << " threads." << std::endl;
          std::vector<std::thread> threads;
          for (unsigned int i=0; i<n_threads; ++i)
            {
              const double omega = MaterialParameters::frequencies[i];
              threads.emplace_back (std::thread ([=] () { do_one_frequency (omega); } ));
            }

          // Now wait for it all:
          for (auto &thread : threads)
            thread.join();
        }
      
      logger << "INFO Number of frequencies computed: "
                << results.size() << std::endl;      

      // Whether or not a termination signal has been sent, try to
      // remove the file that indicates this signal. That's because if
      // we don't do that, the next call to this program won't produce
      // anything at all.
      std::remove ((instance_folder + "/termination_signal").c_str());
    }
  catch (std::exception &exc)
    {
      logger << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      logger << "ERROR Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      logger << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      logger << "ERROR Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}
