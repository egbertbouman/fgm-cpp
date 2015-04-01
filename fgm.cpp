#include <map>  
#include <string>
#include <iostream>
#include <Eigen/Dense>

#include <boost/python/to_python_converter.hpp>
#include <boost/python/extract.hpp>
#include <boost/python/module.hpp>
#include <boost/python/dict.hpp>
#include <boost/python/def.hpp>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>


namespace python = boost::python;

namespace {

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

    struct eigen2numpy {
        eigen2numpy()
        {
            python::to_python_converter<Eigen::MatrixXd, eigen2numpy>();
        }

        static PyObject* convert(const Eigen::MatrixXd& m)
        {
            npy_intp shape[] = {m.rows(), m.cols()};
            PyArrayObject* array = (PyArrayObject*) PyArray_SimpleNew(2, shape, NPY_DOUBLE);
            memcpy(PyArray_DATA(array), m.data(), m.size() * sizeof(double));
            return (PyObject*)array;
        }
    };

    struct numpy2eigen {
        numpy2eigen()
        {
            python::converter::registry::push_back(&convertible, &construct, python::type_id<Eigen::MatrixXd>());
        }

        static void* convertible(PyObject* po)
        {
            return PyArray_Check(po) ? po : 0;
        }

        static void construct(PyObject* po, python::converter::rvalue_from_python_stage1_data* data)
        {
            PyArrayObject* pao = (PyArrayObject*) po;
            if (!PyArray_ISFLOAT(pao))
                throw std::invalid_argument("PyObject is not an array of floats/doubles!");
            
            void* storage = ((python::converter::rvalue_from_python_storage<Eigen::MatrixXd>*)(data))->storage.bytes;

            npy_intp* shape = PyArray_SHAPE(pao);
            Eigen::MatrixXd* m = new (storage) Eigen::MatrixXd(shape[0], shape[1]);
            memcpy(m->data(), PyArray_DATA(pao), m->size() * sizeof(double));
            data->convertible = storage;
        }
    };

    void initializeConverters()
    {
        import_array();
        eigen2numpy();
        numpy2eigen();
    }

    python::object double_matrix_python(PyObject* m)
    {
        Eigen::MatrixXd _m = python::extract<Eigen::MatrixXd>(m);
        _m *= 2;
        return python::object(_m);
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
        return python::object(_Ct);
    }
}


BOOST_PYTHON_MODULE(fgm)
{
    python::def("double_matrix", double_matrix_python);
    python::def("fgm", fgm_python);
    initializeConverters();
}
