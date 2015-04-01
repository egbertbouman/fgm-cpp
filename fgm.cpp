#include <map>  
#include <string>
#include <iostream>
#include <Eigen/Dense>

#include <boost/python/extract.hpp>
#include <boost/python/module.hpp>
#include <boost/python/dict.hpp>
#include <boost/python/def.hpp>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>


namespace python = boost::python;

namespace {

  Eigen::MatrixXd numpy2matrix(const PyObject* po)
  {
    if (!PyArray_Check(po))
      throw std::invalid_argument("PyObject is not an array!");
  
    PyArrayObject* pao = (PyArrayObject*) po;
    if (!PyArray_ISFLOAT(pao))
      throw std::invalid_argument("PyObject is not an array of floats/doubles!");
  
    npy_intp *shape = PyArray_SHAPE(pao);
    Eigen::MatrixXd m(shape[0], shape[1]);
    memcpy(m.data(), PyArray_DATA(pao), m.size() * sizeof(double));
    return m;
  }

  python::object matrix2numpy(const Eigen::MatrixXd& m)
  {
    npy_intp shape[] = {m.rows(), m.cols()};
    PyArrayObject *array = (PyArrayObject*) PyArray_SimpleNew(2, shape, NPY_DOUBLE);
    memcpy(PyArray_DATA(array), m.data(), m.size() * sizeof(double));
    return python::object(python::handle<>((PyObject*) array));
  }
  
  template <class Key, class Value>
  std::map<Key, Value> dict2map(const python::dict &dict)
  {
    std::map<Key, Value> map;
    python::list keys = dict.keys();
    for (int i = 0; i < python::len(keys); i++)
    {
      python::extract<Key> key(keys[i]);
      python::extract<Value> value(dict[(std::string)key]);
      map[key] = value;
    }
    return map;
  }

  template <class Key, class Value>
  python::dict map2dict(const std::map<Key, Value> &map)
  {
    python::dict dict;
    for (const auto &it : map)
      dict[it.first] = it.second;
    return dict;
  }
  
  void init_numpy() { import_array(); }

  python::object double_matrix_python(PyObject* m) {
    init_numpy();
    Eigen::MatrixXd _m = numpy2matrix(m);
    return matrix2numpy(_m*2);
  }
}


BOOST_PYTHON_MODULE(fgm)
{
    python::def("double_matrix", double_matrix_python);
}
