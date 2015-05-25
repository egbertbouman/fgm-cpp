#include <map>  
#include <string>
#include <Eigen/Dense>

#include <boost/python/to_python_converter.hpp>
#include <boost/python/extract.hpp>
#include <boost/python/dict.hpp>
#include <boost/python/def.hpp>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

namespace python = boost::python;


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

        int rows = PyArray_DIMS(pao)[0];
        int cols = PyArray_DIMS(pao)[1];
        Eigen::MatrixXd& m = * new (storage) Eigen::MatrixXd(rows, cols);
        Eigen::MatrixXd::Scalar* pyData = (Eigen::MatrixXd::Scalar*) PyArray_DATA(pao);
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                m(i, j) = pyData[i * cols + j];

        data->convertible = storage;
    }
};

void initializeConverters()
{
    import_array();
    eigen2numpy();
    numpy2eigen();
}
