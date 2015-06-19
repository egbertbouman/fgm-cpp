#include "fgm.hpp"
#include "converters.cpp"

#include <boost/python/dict.hpp>
#include <boost/python/module.hpp>
#include <boost/python/extract.hpp>
#include <boost/python/scope.hpp>
#include <boost/python/args.hpp>

namespace python = boost::python;


python::object solve_python(PyObject* KP, PyObject* KQ, PyObject* Ct, PyObject* asgTX,
                          const python::dict& gph1, const python::dict& gph2,
                          int nAlp, int nItMa, int nHst)
{
    Eigen::MatrixXd _KP = python::extract<Eigen::MatrixXd>(KP);
    Eigen::MatrixXd _KQ = python::extract<Eigen::MatrixXd>(KQ);
    Eigen::MatrixXd _Ct = python::extract<Eigen::MatrixXd>(Ct);
    Eigen::MatrixXd _asgTX = python::extract<Eigen::MatrixXd>(asgTX);
    auto _gph1 = dict2map<std::string, Eigen::MatrixXd>(gph1);
    auto _gph2 = dict2map<std::string, Eigen::MatrixXd>(gph2);
    return python::object(fgm(_KP, _KQ, _Ct, _asgTX, _gph1, _gph2, nAlp, nItMa, nHst));
}


BOOST_PYTHON_MODULE(fgm)
{
    initializeConverters();
    python::def("solve", solve_python,  (python::arg("nAlp")=101, python::arg("nItMa")=100, python::arg("nHst")=10));
    python::scope().attr("version") = FGM_VERSION;
}
