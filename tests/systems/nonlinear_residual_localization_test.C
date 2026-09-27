// The libMesh Finite Element Library.
// Copyright (C) 2002-2026 Benjamin S. Kirk, John W. Peterson, Roy H. Stogner

// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.

// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.

// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

#include <libmesh/libmesh_config.h>

// The residual callbacks exist only with PETSc.
#ifdef LIBMESH_HAVE_PETSC

#include <libmesh/dof_map.h>
#include <libmesh/equation_systems.h>
#include <libmesh/int_range.h>
#include <libmesh/mesh.h>
#include <libmesh/mesh_generation.h>
#include <libmesh/nonlinear_implicit_system.h>
#include <libmesh/nonlinear_solver.h>
#include <libmesh/parallel_implementation.h> // max()
#include <libmesh/petsc_nonlinear_solver.h>
#include <libmesh/petsc_vector.h>

#include "test_comm.h"
#include "libmesh_cppunit.h"

using namespace libMesh;

namespace
{
// Counts System::update(), through which every residual callback localizes x.
class CountingSystem : public NonlinearImplicitSystem
{
public:
  CountingSystem(EquationSystems & es, const std::string & name, const unsigned int number) :
      NonlinearImplicitSystem(es, name, number) {}

  virtual void update() override
  {
    ++n_updates;
    NonlinearImplicitSystem::update();
  }

  unsigned int n_updates = 0;
};

// x^3 + x - 1: the residual below and the values the test expects both call this
// one function, so that both round the same way.
Number cubic(Number x) { return x * x * x + x - Number(1); }

// R_i = cubic(x_i) on current_local_solution, which is what the callbacks pass.
// Records the value of DOF 1 it reads, so that the test sees whether the exact
// constraint enforcement ran before the residual.
class CubicResidual : public NonlinearImplicitSystem::ComputeResidual
{
public:
  virtual void residual(const NumericVector<Number> & X,
                        NumericVector<Number> & R,
                        NonlinearImplicitSystem &) override
  {
    for (auto i : make_range(R.first_local_index(), R.last_local_index()))
      {
        if (i == 1)
          dof1_seen = X(i);
        R.set(i, cubic(X(i)));
      }
    R.close();
  }

  // X(1) as the residual read it, on the processor that owns DOF 1; 0 elsewhere.
  Number dof1_seen = 0;
};

#ifdef LIBMESH_ENABLE_CONSTRAINTS
// u_1 = u_0, added on the processor that owns DOF 1.
class CopyConstraint : public System::Constraint
{
public:
  CopyConstraint(System & sys) : _sys(sys) {}

  virtual void constrain() override
  {
    DofMap & dof_map = _sys.get_dof_map();
    if (dof_map.first_dof() <= 1 && 1 < dof_map.end_dof())
      {
        DofConstraintRow row;
        row[0] = 1.;
        dof_map.add_constraint_row(1, row);
      }
  }

private:
  System & _sys;
};
#endif // LIBMESH_ENABLE_CONSTRAINTS
}

class NonlinearResidualLocalizationTest : public CppUnit::TestCase
{
public:
  LIBMESH_CPPUNIT_TEST_SUITE( NonlinearResidualLocalizationTest );
  CPPUNIT_TEST( testUnconstrainedLocalizesOnce );
#ifdef LIBMESH_ENABLE_CONSTRAINTS
  CPPUNIT_TEST( testConstrainedKeepsSecondLocalization );
  CPPUNIT_TEST( testInexactEnforcementLocalizesOnce );
#endif
  CPPUNIT_TEST_SUITE_END();

public:
  void setUp() {}
  void tearDown() {}

  void testUnconstrainedLocalizesOnce() { LOG_UNIT_TEST; run(false, true, 1); }
#ifdef LIBMESH_ENABLE_CONSTRAINTS
  void testConstrainedKeepsSecondLocalization() { LOG_UNIT_TEST; run(true, true, 2); }
  void testInexactEnforcementLocalizesOnce() { LOG_UNIT_TEST; run(true, false, 1); }
#endif

private:
  static bool passes_everywhere(bool passes_here)
  {
    unsigned int failed = passes_here ? 0 : 1;
    TestCommWorld->max(failed);
    return failed == 0;
  }

  void run(bool with_constraint, bool exact, unsigned int updates_per_call)
  {
    Mesh mesh(*TestCommWorld);
    MeshTools::Generation::build_line(mesh, 9, 0., 1., EDGE2);
    EquationSystems es(mesh);
    CountingSystem & sys = es.add_system<CountingSystem>("nl");
    sys.add_variable("u", CONSTANT, MONOMIAL);
#ifdef LIBMESH_ENABLE_CONSTRAINTS
    CopyConstraint constraint(sys);
    if (with_constraint)
      sys.attach_constraint_object(constraint);
#endif
    es.init();

    CubicResidual residual;
    sys.nonlinear_solver->residual_object = &residual;
    auto & solver = cast_ref<PetscNonlinearSolver<Number> &>(*sys.nonlinear_solver);
    solver.set_exact_constraint_enforcement(exact);
    SNES snes = solver.snes(); // calls init()

    std::unique_ptr<NumericVector<Number>> x = sys.solution->clone();
    for (auto i : make_range(x->first_local_index(), x->last_local_index()))
      x->set(i, Number(0.1) * Number(i + 1));
    x->close();
    std::unique_ptr<NumericVector<Number>> r = sys.rhs->clone();
    Vec xv = cast_ptr<PetscVector<Number> *>(x.get())->vec();
    Vec rv = cast_ptr<PetscVector<Number> *>(r.get())->vec();

    // x_0, which the constraint residual of DOF 1 reads, from the owner of DOF 0.
    Number x0 = (x->first_local_index() == 0) ? (*x)(0) : Number(0);
    TestCommWorld->sum(x0);
    // x_1, from the owner of DOF 1.
    Number x1 = (x->first_local_index() <= 1 && 1 < x->last_local_index()) ? (*x)(1) : Number(0);
    TestCommWorld->sum(x1);
    // What the residual must read at DOF 1: x_0 once the exact enforcement ran, x_1 otherwise.
    const Number dof1_expected = (with_constraint && exact) ? x0 : x1;

    using Callback = PetscErrorCode (*)(SNES, Vec, Vec, void *);
    for (Callback callback : {libmesh_petsc_snes_residual,
                              libmesh_petsc_snes_fd_residual,
                              libmesh_petsc_snes_mffd_residual})
      {
        const unsigned int before = sys.n_updates;
        residual.dof1_seen = 0;
        LibmeshPetscCallA(TestCommWorld->get(), callback(snes, xv, rv, &solver));
        CPPUNIT_ASSERT(passes_everywhere(sys.n_updates - before == updates_per_call));

        // The value of DOF 1 the residual read, from its owner; the same sum on every processor.
        Number dof1_seen = residual.dof1_seen;
        TestCommWorld->sum(dof1_seen);
        CPPUNIT_ASSERT(passes_everywhere(dof1_seen == dof1_expected));

        bool values_ok = true;
        for (auto i : make_range(r->first_local_index(), r->last_local_index()))
          {
            const Number expected = (with_constraint && exact && i == 1) ? (*x)(1) - x0 : cubic((*x)(i));
            values_ok = values_ok && ((*r)(i) == expected);
            values_ok = values_ok && ((*sys.current_local_solution)(i) == (*x)(i));
          }
        CPPUNIT_ASSERT(passes_everywhere(values_ok));
      }
  }
};

CPPUNIT_TEST_SUITE_REGISTRATION( NonlinearResidualLocalizationTest );

#endif // LIBMESH_HAVE_PETSC
