// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.
//
// Created on: Jan 22, 2019
// Author: William F Godoy godoywf@ornl.gov
// adios2: Adaptable Input/Output System https://github.com/ornladios/ADIOS2

#ifndef MFEM_ADIOS2STREAM
#define MFEM_ADIOS2STREAM

#include "../config/config.hpp"

#include <map>
#include <memory>  // std::shared_ptr
#include <string>
#include <set>

#ifdef MFEM_USE_MPI
#include <mpi.h>
#endif

#include <adios2.h>

namespace mfem
{

// forward declaring classes to avoid circular references
class Vector;
class GridFunction;
class Mesh;

template <class T>
class Array;
class Element;

class adios2stream
{
   friend class Vector;
   friend class GridFunction;
   friend class Mesh;

public:
   /**
    * Open modes for adios2stream (from std::fstream)
    * out: write
    * in:  read
    * app: append
    */
   enum class openmode { out, in, app };

   /** Print and Save modes, deferred is done at Close or EndStep, sync is immediate */
   enum class mode {sync, deferred};

   enum class data_type {none, point_data, cell_data};

#ifdef MFEM_USE_MPI

   /**
    * adios2stream MPI constructor, allows for passing parameters in source
    * code (compile-time) only.
    * @param name stream name
    * @param mode adios2stream::openmode::in (Read), adios2stream::openmode::out
    * (Write)
    * @param comm MPI communicator establishing domain for fstream
    * @param engineType available adios2 engine, default is BP3 file
    *        see https://adios2.readthedocs.io/en/latest/engines/engines.html
    * @throws std::invalid_argument (user input error) or std::runtime_error
    *         (system error)
    */
   adios2stream(const std::string& name, const openmode mode, MPI_Comm comm,
                const std::string engine_type = "BPFile");
#else
   /**
    * adios2stream Non-MPI serial constructor, allows for passing parameters in
    * source code (compile-time) only.
    * @param name stream name
    * @param mode adios2stream::openmode::in (Read), adios2stream::openmode::out
    * (Write)
    * @param engineType available adios2 engine, default is BP3 file
    * @throws std::invalid_argument (user input error) or std::runtime_error
    *         (system error)
    */
   adios2stream(const std::string& name, const openmode mode,
                const std::string engine_type = "BPFile");
#endif

   /** calls Close if stream is valid basically follows C++ RAII **/
   virtual ~adios2stream();

   /**
    * Set parameters for a particular adios2stream Engine
    * See https://adios2.readthedocs.io/en/latest/engines/engines.html
    * @param parameters map of key/value string elements
    */
   void SetParameters(const std::map<std::string, std::string>& parameters =
                         std::map<std::string, std::string>());

   /**
    * Single parameter version of SetParameters passing a key/value pair
    * See https://adios2.readthedocs.io/en/latest/engines/engines.html
    * @param key input parameter key
    * @param value input parameter value
    */
   void SetParameter(const std::string key, const std::string value);

   /** Begins an I/O step */
   void BeginStep();

   /** Ends the current step, by default transports the data */
   void EndStep();

   /**
       * Associates a physical time with the current I/O step
       * input physical time parameter.
       * @param time input physical time
       * @param variable_name variable name associated with time
       */
   void SetTime(const double time, const std::string& variable_name = "TIME");

   /** Return the current step between BeginStep and EndStep */
   size_t CurrentStep() const;

   /** Finished interaction with adios2stream and flushes the data */
   void Close();

protected:

   /**
    * Called from friend class Mesh (which is called from ParMesh)
    * @param mesh input Mesh object to print
    * @param print_mode sync: one at a time, deferred: collected (pre-fetch)
    */
   void Print(const Mesh& mesh, const adios2stream::mode print_mode = mode::sync);

   void Save(const GridFunction& grid_function, const std::string& variable_name,
             const data_type type);

private:
   /** placeholder for engine name */
   const std::string name;

   /** placeholder for engine openmode */
   const openmode adios2_openmode;

   /** main adios2 object that owns all the io and engine components */
   std::shared_ptr<adios2::ADIOS> adios;

   /** io object to set parameters, variables and engines */
   adios2::IO io;

   /** heavy object doing system-level I/O operations */
   adios2::Engine engine;

   /** true : engine step is active after engine.BeginStep(),
     *  false: inactive after engine.EndStep() */
   bool active_step = false;

   /** true: mesh is defined, false: not yet */
   bool is_mesh_defined = false;

   /** ordering of the nodes to be passed to the schema as an attribute
    *  true: XXX YYY ZZZ, false: XYZ, XYZ, XYZ
    *  if true it must swap the vertices to Ordering::byDIM*/
   bool ordering_by_node = false;

   /** true: refine solution at Save */
   bool refine = true;

   /** refinement level at Save and Print */
   int refinement_level = 1;

   /** save for point data */
   size_t refined_mesh_nvertices = 0;

   /** save for cell data */
   size_t refined_mesh_nelements = 0;

   /** saves the variable names representing point data */
   std::set<std::string> point_data_variables;

   /** stores the value of the physical time variable from SetTime */
   std::string time_variable_name = "";

   /**
    * Map glvis element types to VTK element types
    * @param glvisType input
    * @return VTK element type
    */
   int32_t GLVISToVTKType(const int glvisType) const noexcept;

   /** sets the current vtk_schema from point data arrays to be parsed
    *  in VTK for Paraview visualization */
   std::string VTKSchema() const noexcept;

   /**
    * Checks if array of elements contains onlu contant types
    * @param elements array input to check
    * @return true: types are constance, false: mixed types
    */
   bool IsConstantElementType(const Array<Element*>& elements ) const noexcept;

   /**
    * Maps to appropriate adios2::Mode from out, in to write, read
    * @param mode
    * @return
    */
   adios2::Mode ToADIOS2Mode(const adios2stream::openmode mode) const noexcept;

};

}  // end namespace mfem

#endif /* MFEM_ADIOS2STREAM */
