#include "fgm.cpp"
#include "converters.cpp"

#include <boost/python/dict.hpp>
#include <boost/python/module.hpp>
#include <boost/python/extract.hpp>

namespace python = boost::python;


python::object double_matrix_python(PyObject* m)
{
    Eigen::MatrixXd _m = python::extract<Eigen::MatrixXd>(m);
    return python::object(double_matrix(_m));
}

python::object fgm_python(PyObject* KP, PyObject* KQ, PyObject* Ct, PyObject* asgTX,
                          const python::dict& gph1, const python::dict& gph2,
                          const python::dict& params)
{
    Eigen::MatrixXd _KP = python::extract<Eigen::MatrixXd>(KP);
    Eigen::MatrixXd _KQ = python::extract<Eigen::MatrixXd>(KQ);
    Eigen::MatrixXd _Ct = python::extract<Eigen::MatrixXd>(Ct);
    Eigen::MatrixXd _asgTX = python::extract<Eigen::MatrixXd>(asgTX);
    auto _gph1 = dict2map<std::string, Eigen::MatrixXd>(gph1);
    auto _gph2 = dict2map<std::string, Eigen::MatrixXd>(gph2);
    auto _params = dict2map<std::string, std::string>(params);
    return python::object(fgm(_KP, _KQ, _Ct, _asgTX, _gph1, _gph2, _params));
}


BOOST_PYTHON_MODULE(fgm)
{
    initializeConverters();
    python::def("double_matrix", double_matrix_python);
    python::def("fgm", fgm_python);
}