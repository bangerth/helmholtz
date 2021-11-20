/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2021 by Wolfgang Bangerth and SAATI Co.
 *
 * ---------------------------------------------------------------------
 */


#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/function_lib.h>
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
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/data_out_faces.h>
#include <deal.II/numerics/vector_tools_point_value.h>
#include <deal.II/numerics/vector_tools_point_gradient.h>

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
#include <regex>


std::string instance_folder;
std::string output_file_prefix;
  
std::ofstream logger;

namespace TransmissionProblem
{
  using namespace dealii;

  using ScalarType = std::complex<double>;

  // The following namespace defines material parameters. We use SI units.
  namespace MaterialParameters
  {
    std::vector<std::pair<double,unsigned int>> frequencies;

    std::unique_ptr<Functions::InterpolatedTensorProductGridData<1>> density_real;
    std::unique_ptr<Functions::InterpolatedTensorProductGridData<1>> density_imag;

    std::unique_ptr<Functions::InterpolatedTensorProductGridData<1>> bulk_modulus_real;
    std::unique_ptr<Functions::InterpolatedTensorProductGridData<1>> bulk_modulus_imag;
  }

  std::string mesh_file_name;
  double geometry_conversion_factor;

  unsigned int fe_degree = 2;
  int n_mesh_refinement_steps = 5;

  unsigned int n_threads = 0;

  std::vector<Point<3>> evaluation_points;


  void
  declare_parameters (ParameterHandler &prm)
  {
    prm.declare_entry ("Mesh file name", "./mesh.msh",
                       Patterns::FileName(),
                       "The name of the file from which to read the mesh. "
                       "The extents of the geometry so described are scaled "
                       "by the 'Geometry conversion factor to SI units' parameter.");
    prm.declare_entry ("Geometry conversion factor to meters", "1",
                       Patterns::Double(),
                       "A conversion factor from whatever length units are used "
                       "for the geometry description and for the description of "
                       "arbitrary evaluation points, to meters. For example, if the mesh and "
                       "the evaluation points are given in multiples of inches, "
                       "then this factor should be set to 0.0254 because one inch "
                       "equals 0.0254 meters = 25.4 mm.");

    prm.declare_entry ("Material properties file name",
                       "./material_properties.txt",
                       Patterns::FileName(Patterns::FileName::input),
                       "The name of the file from which to read the mechanical material response.");

    prm.declare_entry ("Evaluation points", "",
                       Patterns::List (Patterns::List(Patterns::Double(),3,3,","),
                                       0, Patterns::List::max_int_value, ";"),
                       "A list of points at which the program evaluates both the pressure "
                       "and the (volumetric) velocity. Each point is specified by x,y,z "
                       "coordinates using the same units as used for the mesh (and then "
                       "scaled by the value of the 'Geometry conversion factor to meters' "
                       "parameter). Points are separated by semicolons.");

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

    // Start with geometry things: The mesh, the scaling factor, evaluation points
    mesh_file_name = prm.get ("Mesh file name");
    geometry_conversion_factor = prm.get_double ("Geometry conversion factor to meters");
    {
      const auto points = Utilities::split_string_list (prm.get("Evaluation points"), ';');
      for (const auto point : points)
        {
          const auto coordinates = Utilities::split_string_list (point, ',');
          AssertDimension (coordinates.size(), 3);

          Point<3> p;
          for (unsigned int d=0; d<3; ++d)
            p[d] = Utilities::string_to_double(coordinates[d]);

          p *= geometry_conversion_factor;

          evaluation_points.push_back (p);
        }
    }


    // Next up: Material parameters.
    const std::string material_properties_file_name
      = prm.get ("Material properties file name");
    {
      std::ifstream material_input (material_properties_file_name);
      AssertThrow (material_input, ExcIO());

      // discard the first line as a comment
      {
        std::string dummy;
        std::getline(material_input, dummy);
      }

      std::vector<double> frequencies;
      std::vector<double> rho_real;
      std::vector<double> rho_imag;
      std::vector<double> K_real;
      std::vector<double> K_imag;

      // Read until we find the end of the file
      while (true)
        {
          double f, rr, ri, Kr, Ki;
          material_input >> f >> rr >> ri >> Kr >> Ki;

          if (material_input.good())
            {
              frequencies.push_back (f);
              rho_real.push_back (rr);
              rho_imag.push_back (ri);
              K_real.push_back (Kr);
              K_imag.push_back (Ki);
            }
          else
            break;
        }

      logger << "INFO Read material parameters for "
             << frequencies.size()
             << " frequencies ranging from "
             << frequencies.front()
             << " to "
             << frequencies.back()
             << "Hz."
             << std::endl;

      // Now convert these arrays into Table objects as desired by the
      // InterpolatedTensorProductGridData class
      Table<1,double> rr (frequencies.size(), rho_real.begin());
      Table<1,double> ri (frequencies.size(), rho_imag.begin());
      Table<1,double> Kr (frequencies.size(), K_real.begin());
      Table<1,double> Ki (frequencies.size(), K_imag.begin());

      // And finally use these to initialize the
      // InterpolatedTensorProductGridData objects themselves
      const std::array<std::vector<double>,1> f({{frequencies}});
      MaterialParameters::density_real
        = std::make_unique<Functions::InterpolatedTensorProductGridData<1>>(f, rr);
      MaterialParameters::density_imag
        = std::make_unique<Functions::InterpolatedTensorProductGridData<1>>(f, ri);
      MaterialParameters::bulk_modulus_real
        = std::make_unique<Functions::InterpolatedTensorProductGridData<1>>(f, Kr);
      MaterialParameters::bulk_modulus_imag
        = std::make_unique<Functions::InterpolatedTensorProductGridData<1>>(f, Ki);
    }

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
        unsigned int index = 0;
        for (double omega = min_omega;
             omega <= max_omega;
             omega += delta_omega, ++index)
          MaterialParameters::frequencies.emplace_back (omega,index);
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
        unsigned int index = 0;
        for (double log_omega = log_min_omega;
             log_omega <= log_max_omega;
             log_omega += delta_log_omega, ++index)
          MaterialParameters::frequencies.emplace_back (std::exp(log_omega),index);
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
        const std::vector<double> freq_list =
          Utilities::string_to_double
          (Utilities::split_string_list
           (Utilities::trim (parenthesized_expr.substr
                             (1,
                              parenthesized_expr.size() - 2)),
            ','));
        AssertThrow (freq_list.size() >= 1,
                     ExcMessage ("Wrong format for 'list' frequency spacing."));

        for (unsigned int i=0; i<freq_list.size(); ++i)
          MaterialParameters::frequencies.emplace_back (freq_list[i], i);

        
        // Because MaterialParameters::frequencies stores angular
        // frequencies, we need to multiply by 2*pi
        for (auto &f : MaterialParameters::frequencies)
          f.first *= 2 * numbers::PI;
      }
    else
      AssertThrow (false,
                   ExcMessage ("The format for the description of the frequencies to "
                               "be solved for, namely <"
                               + frequency_descriptor + ">, did not match any of "
                               "the recognized formats."));

    fe_degree               = prm.get_integer ("Finite element polynomial degree");
    n_mesh_refinement_steps = prm.get_integer ("Number of mesh refinement steps");

    n_threads = prm.get_integer ("Number of threads");
  }





  // A data structure that is used to collect the results of the computations
  // for one frequency. The main class fills this for a given frequency
  // in various places of its member functions, and at the end puts it into
  // a global map.
  struct OutputData
  {
    // For each source port (first index), store integrated pressures
    // and velocities at all other ports (second index)
    FullMatrix<ScalarType> P, U;

    // For each source port (first index), store pressure and velocity
    // at all evaluation points (second index)
    Table<2,std::complex<double>>             evaluation_point_pressures;
    Table<2,Tensor<1,3,std::complex<double>>> evaluation_point_velocities;


    std::vector<std::string> visualization_file_names;
  };



  TimerOutput timer_output = TimerOutput (logger, TimerOutput::summary,
                                          TimerOutput::wall_times);


  // Create a file that indicates that program execution failed for
  // some reason. Then exit the program with an error code.
  void create_failure_file_and_exit ()
  {
    {
      std::ofstream failure_signal (instance_folder + "/" +
                                    output_file_prefix +
                                    "solver_failure_signal.txt");
      failure_signal << "FAILURE" << std::endl;
    }
    
    std::exit(1);
  }
  
  
  // Check whether an external program has left a signal that
  // indicates that the current program run should terminate without
  // computing any further frequency responses. This is done by
  // placing the word "STOP" into the file "termination_signal" in the
  // current directory.
  //
  // Once detected, we delete the file again and terminate the
  // program.
  void check_for_termination_signal()
  {
    // Try and see whether we can open the file at all. If we can't,
    // then no termination signal has been sent. If so, return 'true',
    // but before that set a flag that ensures we don't have to do the
    // expensive test with the file in any further calls. (We'll try
    // to abort the program below, but this may block for a bit
    // because we need to wait for the lock that guards access to the
    // output file.)
    std::ifstream in(instance_folder + "/" +
                     output_file_prefix +
                     "termination_signal.txt");
    if (!in)
      return;

    // OK, the file exists, but does it contain the right content?
    std::string line;
    std::getline(in, line);
    if (line == "STOP")
      {
        // Close the file handle and remove the file.
        in.close();
        std::remove ((instance_folder + "/" +
                      output_file_prefix +
                      "termination_signal.txt").c_str());

        logger << "INFO *** Terminating program upon request." << std::endl;
        create_failure_file_and_exit();
      }

    // The file exists, but it has the wrong content (or no content so
    // far). This means no termination. In the best of all cases, we
    // will have caught the driver program having created but not
    // written to the file. The next time we check, we might find the
    // file in the correct state.
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
    HelmholtzProblem(const double omega, const unsigned int frequency_number);

    void run();

  private:
    void make_grid();
    void setup_system();
    void assemble_system(const unsigned int current_source_port);
    void solve();
    void postprocess(const unsigned int current_source_port);
    void output_results(const unsigned int current_source_port);
    void output_evaluated_data ();
    
    // The frequency that this instance of the class is supposed to
    // solve for, its number in the list, along with the density and
    // wave speed to be used for this frequency.
    const double               omega;
    const unsigned int         frequency_number;

    const std::complex<double> density;
    const std::complex<double> wave_speed;


    Triangulation<dim>                   triangulation;
    std::vector<types::boundary_id>      port_boundary_ids;
    std::vector<double>                  port_areas;

    std::unique_ptr<Mapping<dim>>        mapping;

    std::unique_ptr<FiniteElement<dim>> fe;
    DoFHandler<dim>                     dof_handler;

    SparsityPattern                     sparsity_pattern;
    SparseMatrix<ScalarType>            system_matrix;

    Vector<ScalarType>                  solution;
    Vector<ScalarType>                  system_rhs;

    OutputData                          output_data;
  };



  template <int dim>
  HelmholtzProblem<dim>::HelmholtzProblem(const double omega,
                                          const unsigned int frequency_number)
    : omega (omega)
      , frequency_number(frequency_number)
      , density (MaterialParameters::density_real->value(Point<1>(omega)),
                 MaterialParameters::density_imag->value(Point<1>(omega)))
      , wave_speed (std::sqrt(std::complex<double>(MaterialParameters::bulk_modulus_real->value(Point<1>(omega)),
                                                   MaterialParameters::bulk_modulus_imag->value(Point<1>(omega)))
                              / density))
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

    // Determine what format we want to read the mesh in: .mphtxt =>
    // COMSOL; .msh => GMSH
    if (std::regex_match(mesh_file_name,
                         std::regex(".*\\.mphtxt", std::regex_constants::basic)))
      {
        logger << "Reading <" << mesh_file_name
               << "> in COMSOL .mphtxt format" << std::endl;
        grid_in.read_comsol_mphtxt (input);
      }
    else if (std::regex_match(mesh_file_name,
                              std::regex(".*\\.msh", std::regex_constants::basic)))
      {
        logger << "Reading <" << mesh_file_name
               << "> in GMSH .msh format" << std::endl;
        grid_in.read_msh (input);
      }
    else if (std::regex_match(mesh_file_name,
                              std::regex(".*\\.inp", std::regex_constants::basic)))
      {
        logger << "Reading <" << mesh_file_name
               << "> in ABAQUS .inp format" << std::endl;
        grid_in.read_abaqus (input);
      }
    else
      AssertThrow (false,
                   ExcMessage ("The file ending for the mesh file <"
                               + mesh_file_name +
                               "> is not supported."));

    std::cout << "The mesh has " << triangulation.n_active_cells() << " cells" << std::endl;
    // Scale the triangulation by the geometry factor
    GridTools::scale (geometry_conversion_factor, triangulation);

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


    // Depending on what reference cell the triangulation uses, pick
    // the right finite element
    Assert (triangulation.get_reference_cells().size() == 1,
            ExcMessage("We don't know how to deal with this kind of mesh."));
    if (triangulation.get_reference_cells()[0] == ReferenceCells::Tetrahedron)
      {
        fe = std::make_unique<FE_SimplexP<dim>>(TransmissionProblem::fe_degree);
        mapping = ReferenceCells::Tetrahedron.get_default_mapping<dim,dim>(1);
      }
    else if (triangulation.get_reference_cells()[0] == ReferenceCells::Hexahedron)
      {
        fe = std::make_unique<FE_Q<dim>>(TransmissionProblem::fe_degree);
        mapping = ReferenceCells::Hexahedron.get_default_mapping<dim,dim>(1);
      }
    else
      Assert (false, ExcMessage("We don't know how to deal with this kind of mesh."));


    // Finally output the mesh with ports correctly colored
    {
      class BoundaryIds : public DataPostprocessorScalar<dim>
      {
    public:
      BoundaryIds()
      : DataPostprocessorScalar<dim>("boundary_id", update_quadrature_points)
          {}


      virtual void evaluate_scalar_field(
        const DataPostprocessorInputs::Scalar<dim> &inputs,
        std::vector<Vector<double>> &computed_quantities) const override
          {
            AssertDimension(computed_quantities.size(),
                            inputs.solution_values.size());

            const typename DoFHandler<dim>::active_cell_iterator
              cell = inputs.template get_cell<dim>();
            
            // First find out which face the quadrature points belong to.
            unsigned int face=0;
            for (; face<cell->n_faces(); ++face)
              {
                bool vertices_match = true;
                for (const unsigned int v : cell->face(face)->vertex_indices())
                  if (cell->face(face)->vertex(v).distance(inputs.evaluation_points[v]) > 1e-12)
                    vertices_match = false;

                if (vertices_match == true)
                  break;
              }
            Assert (face != cell->n_faces(), ExcInternalError());
            
            for (unsigned int q = 0; q < computed_quantities.size(); ++q)
              {
                AssertDimension(computed_quantities[q].size(), 1);
                
                computed_quantities[q](0) = cell->face(face)->boundary_id();
              }
          }
      };

      BoundaryIds boundary_ids;
        
      
      DataOutFaces<dim> data_out_faces;

      std::unique_ptr<FiniteElement<dim>> fe;
      Assert (triangulation.get_reference_cells().size() == 1,
              ExcMessage("We don't know how to deal with this kind of mesh."));
      if (triangulation.get_reference_cells()[0] == ReferenceCells::Tetrahedron)
        fe = std::make_unique<FE_SimplexP<dim>>(TransmissionProblem::fe_degree);
      else if (triangulation.get_reference_cells()[0] == ReferenceCells::Hexahedron)
        fe = std::make_unique<FE_Q<dim>>(TransmissionProblem::fe_degree);
      else
        Assert (false, ExcMessage("We don't know how to deal with this kind of mesh."));

      DoFHandler<dim> dummy_dof_handler(triangulation);
      dummy_dof_handler.distribute_dofs(*fe);

      Vector<double> dummy_solution (dummy_dof_handler.n_dofs());
      
      data_out_faces.attach_dof_handler(dummy_dof_handler);
      data_out_faces.add_data_vector(dummy_solution, boundary_ids);
      data_out_faces.build_patches();

      std::ofstream out("surface.vtu");
      data_out_faces.write_vtu(out);
    }
    
    
    // Figure out what boundary ids we have that describe ports. We
    // take these as all of those boundary ids that are non-zero
    port_boundary_ids = triangulation.get_boundary_ids();
    for (const auto &id : port_boundary_ids)
      logger << "Found boundary id " << id << std::endl;
    
    port_boundary_ids.erase (std::find(port_boundary_ids.begin(),
                                       port_boundary_ids.end(),
                                       types::boundary_id(0)));

    // Now also correctly size the matrices we compute for each
    // frequency
    output_data.P.reinit (port_boundary_ids.size(),
                          port_boundary_ids.size());
    output_data.U.reinit (port_boundary_ids.size(),
                          port_boundary_ids.size());

    output_data.evaluation_point_pressures.reinit (port_boundary_ids.size(),
                                                   evaluation_points.size());
    output_data.evaluation_point_velocities.reinit (port_boundary_ids.size(),
                                                    evaluation_points.size());
    
    
    // As a final step, compute the areas of the various ports so we
    // can later normalize when computing average pressures and
    // velocities:
    port_areas.resize (port_boundary_ids.size(), 0.);
    const QGauss<dim-1>  quadrature_formula(fe->degree + 2);
    const unsigned int n_q_points = quadrature_formula.size();
    FEFaceValues<dim> fe_face_values(*mapping,
                                     *fe,
                                     quadrature_formula,
                                     update_JxW_values);
    for (const auto cell : triangulation.active_cell_iterators())
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

    dof_handler.distribute_dofs(*fe);

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
                     wave_speed *
                     wave_speed *
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
                                    *fe,
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

    // Figure out boundary conditions. We want a unit pressure on the
    // source port, and zero pressure on the other ports. Also zero
    // Neumann conditions on the remaining boundary, but we don't need
    // to do anything specific for that.
    std::map<types::global_dof_index,ScalarType> boundary_values;
    VectorTools::interpolate_boundary_values(dof_handler,
                                             port_boundary_ids[current_source_port],
                                             Functions::ConstantFunction<dim,ScalarType>(1),
                                             boundary_values);
    for (const auto b_id : port_boundary_ids)
      if (b_id != port_boundary_ids[current_source_port])
        VectorTools::interpolate_boundary_values(dof_handler,
                                                 b_id,
                                                 Functions::ZeroFunction<dim,ScalarType>(),
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



  // The next function postprocesses the solution. In the current
  // context, this implies computing integrals over the solution at
  // all ports, as well as evaluating the solution at specific points.
  template <int dim>
  void HelmholtzProblem<dim>::postprocess(const unsigned int current_source_port)
  {
    std::unique_ptr<TimerOutput::Scope> timer_section = (n_threads==1 ? std::make_unique<TimerOutput::Scope>(timer_output, "Postprocess") : nullptr);

    // *** Step 1:
    // Compute integrals of the solution
    const QGauss<dim-1>  quadrature_formula(fe->degree + 2);
    const unsigned int n_q_points = quadrature_formula.size();
    FEFaceValues<dim> fe_face_values(*mapping,
                                     *fe,
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
                  // velocity. The volumetric velocity is defined as
                  //   v = -1/(j rho omega) nabla p
                  // with units
                  //
                  // Note that the velocity is computed *into* the
                  // volume, i.e., with the negative outward normal.
                  const auto velocity = -1./(std::complex<double>(0,1)*density*omega)
                                        * solution_gradients[q_point];

                  output_data.U(current_source_port, this_port)
                    +=  (velocity *
                         (-fe_face_values.normal_vector(q_point)) *
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


    // *** Step 2:
    // Evaluate the solution at specific points
    for (unsigned int point_index = 0; point_index<evaluation_points.size(); ++point_index)
      {
        // First evaluate the values
        output_data.evaluation_point_pressures
          [current_source_port][point_index]
          = VectorTools::point_value (*mapping, dof_handler, solution,
                                      evaluation_points[point_index]);

        // Then also the velocities. Recall from above that
        //   v = -1/(j rho omega) nabla p
        output_data.evaluation_point_velocities
          [current_source_port][point_index]
          = VectorTools::point_gradient (*mapping, dof_handler, solution,
                                         evaluation_points[point_index]);
        output_data.evaluation_point_velocities[current_source_port][point_index]
          *= -1./(std::complex<double>(0,1)*density*omega);
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
    data_out.build_patches(fe->degree);

    const std::string file_name = instance_folder + "/" +
                                  output_file_prefix +
                                  "visualization/solution-" +
                                  std::to_string(frequency_number) +
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



  template <int dim>
  void HelmholtzProblem<dim>::output_evaluated_data ()
  {
    std::ostringstream buffer;

    const unsigned int n_port_boundary_ids = port_boundary_ids.size();
    
    const unsigned int field_width = 12;

    // Filter out values that are in essence zero
    for (unsigned int i=0; i<n_port_boundary_ids; ++i)
      for (unsigned int j=0; j<n_port_boundary_ids; ++j)
        {
          if (std::fabs(std::real(output_data.P(i,j))) < 1e-12 * output_data.P.l1_norm())
            output_data.P(i,j).real(0);
          if (std::fabs(std::imag(output_data.P(i,j))) < 1e-12 * output_data.P.l1_norm())
            output_data.P(i,j).imag(0);
        }

    for (unsigned int i=0; i<n_port_boundary_ids; ++i)
      for (unsigned int j=0; j<n_port_boundary_ids; ++j)
        {
          if (std::fabs(std::real(output_data.P(i,j))) < 1e-12 * output_data.P.l1_norm())
            output_data.P(i,j).real(0);
          if (std::fabs(std::imag(output_data.P(i,j))) < 1e-12 * output_data.P.l1_norm())
            output_data.P(i,j).imag(0);
        }


    buffer << "Results for frequency f="
           << omega/2/numbers::PI << ":\n"
           << "==============================\n\n";

    buffer << "P = [\n";
    for (unsigned int i=0; i<n_port_boundary_ids; ++i)
      {
        buffer << "      [";
        for (unsigned int j=0; j<n_port_boundary_ids; ++j)
          buffer << std::setw(field_width) << std::right << std::real(output_data.P(i,j))
                 << (std::imag(output_data.P(i,j)) >= 0 ? '+' : '-')
                 << "j*"
                 << std::setw(field_width) << std::left << std::fabs(std::imag(output_data.P(i,j)))
                 << ' ';
        buffer << "]\n";
      }
    buffer << "]\n";

    buffer << "\n\nU = [\n";
    for (unsigned int i=0; i<n_port_boundary_ids; ++i)
      {
        buffer << "      [";
        for (unsigned int j=0; j<n_port_boundary_ids; ++j)
          buffer << std::setw(field_width) << std::right << std::real(output_data.U(i,j))
                 << (std::imag(output_data.U(i,j)) >= 0 ? '+' : '-')
                 << "j*"
                 << std::setw(field_width) << std::left << std::fabs(std::imag(output_data.U(i,j)))
                 << ' ';
        buffer << "]\n";
      }
    buffer << "]\n";
    buffer << "\n\n\n" << std::flush;


    // Then also output locations, pressures, and velocities at
    // the evaluation points. Output the location in the
    // coordinates originally provided in the output file.
    buffer << "Pressure and velocity at explicitly specified evaluation points:\n"
           << "================================================================\n\n";

    for (unsigned int e=0; e<evaluation_points.size(); ++e)
      for (unsigned int i=0; i<n_port_boundary_ids; ++i)
        {
          buffer << "Point at "
                 << evaluation_points[e] / geometry_conversion_factor
                 << ":  pressure="
                 << output_data.evaluation_point_pressures[i][e]
                 << ", velocity="
                 << output_data.evaluation_point_velocities[i][e]
                 << '\n';
        }

    buffer << "\n\n\n" << std::flush;


    // Now put it all into a file:
    std::cerr << "Writing data to " +instance_folder + "/" +
                                      output_file_prefix +
                                      "_" +
                                      std::to_string(frequency_number) +
                                      "_" +
      "frequency_response.txt" << std::endl;
    std::ofstream frequency_response (instance_folder + "/" +
                                      output_file_prefix +
                                      "_" +
                                      std::to_string(frequency_number) +
                                      "_" +
                                      "frequency_response.txt");
    frequency_response << buffer.str();
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

    output_evaluated_data ();
  }



  void solve_one_frequency (const double omega,
                            const unsigned frequency_number)
  {
    // The main() function has created tasks for all frequencies
    // provided by the caller, but there is the possibility that a
    // higher instance has decided that the program needs to stop
    // doing what it's doing. Check here, as this is the first
    // non-trivial place one ends up when a task executes, whether we
    // are supposed to actually do anything, or should instead stop
    // working on the frequency this task corresponds to.
    check_for_termination_signal();

    try
      {
        HelmholtzProblem<3> helmholtz_problem(omega,frequency_number);
        helmholtz_problem.run();
      }
    catch (const std::exception &exc)
      {
        std::ofstream error_out (instance_folder + "/" +
                                 output_file_prefix +
                                 "error.log");

        error_out << "ERROR Exception while computing for frequency "
                  << omega/2/numbers::PI << ":\n"
                  << exc.what() << std::endl
                  << "Aborting!" << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        TransmissionProblem::create_failure_file_and_exit();
      }
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
      if (argc >= 3)
        output_file_prefix = std::string(argv[2]);
    }
  else
    {
      instance_folder = std::string(".");
    }

  // First remove the success or failure files, should they exist:
  std::remove ((instance_folder + "/" +
                output_file_prefix +
                "success_signal.txt").c_str());
  std::remove ((instance_folder + "/" +
                output_file_prefix +
                "solver_failure_signal.txt").c_str());
  
  logger.open (instance_folder + "/" +
               output_file_prefix +
               "output.log");
  logger << "INFO Program started with argument '"
         << instance_folder << "'"
         << std::endl;

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
      std::remove ((instance_folder + "/" +
                    output_file_prefix +
                    "frequency_response.txt").c_str());

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
          for (const auto freq : MaterialParameters::frequencies)
            tasks.emplace_back (std::async (std::launch::async,
                                            [=]() { solve_one_frequency (freq.first,
                                                                         freq.second); }));

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
          std::vector<std::pair<double,unsigned int>>
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
          std::function<void (double,unsigned int)> do_one_frequency
            = [&] (const double omega,
                   const unsigned int freq_no) {
                solve_one_frequency (omega, freq_no);

                std::pair<double,unsigned int> next_omega = {-1e20,numbers::invalid_unsigned_int};
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
                do_one_frequency (next_omega.first,
                                  next_omega.second);
          };

          // Now start the initial tasks.
          logger << "INFO Using processing with limited number of "
                    << n_threads << " threads." << std::endl;
          std::vector<std::thread> threads;
          for (unsigned int i=0; i<n_threads; ++i)
            {
              const auto omega = MaterialParameters::frequencies[i];
              threads.emplace_back (std::thread ([=] () { do_one_frequency (omega.first,
                                                                            omega.second); } ));
            }

          // Now wait for it all:
          for (auto &thread : threads)
            thread.join();
        }

      // Whether or not a termination signal has been sent, try to
      // remove the file that indicates this signal. That's because if
      // we don't do that, the next call to this program won't produce
      // anything at all.
      std::remove ((instance_folder + "/termination_signal").c_str());

      // Finally also indicate success:
      std::ofstream success_signal (instance_folder + "/" +
                                    output_file_prefix +
                                    "success_signal.txt");
      success_signal << "SUCCESS" << std::endl;
    }
  catch (std::exception &exc)
    {
      std::ofstream error_out (instance_folder + "/" +
                               output_file_prefix +
                               "error.log");
      error_out << "ERROR Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      TransmissionProblem::create_failure_file_and_exit();
    }
  catch (...)
    {
      std::ofstream error_out (instance_folder + "/" +
                               output_file_prefix +
                               "error.log");
      error_out << "ERROR Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      TransmissionProblem::create_failure_file_and_exit();
    }

  return 0;
}
