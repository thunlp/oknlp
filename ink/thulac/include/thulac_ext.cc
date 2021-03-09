#include "thulac.h"
#include <sstream>
#include <fstream>
#include <Python.h>
using std::cin;
using std::cout;
using std::endl;

static PyObject *method_thulac(PyObject *self, PyObject *args) {
    char* raw = NULL;
    char* user_specified_dict_name=NULL;
    char* model_path_char = NULL;
    bool useT2S = false;
    bool seg_only = true;
    bool useFilter = false;
    /* Parse arguments */
     if(!PyArg_ParseTuple(args, "s", &raw)) {
        return NULL;
    }

    THULAC lac;
    lac.init(model_path_char, user_specified_dict_name, seg_only, useT2S, useFilter);
    THULAC_result result;
    //clock_t start = clock();

    lac.cut(raw, result);
    //clock_t end = clock();
    //double duration = (double)(end - start) / CLOCKS_PER_SEC;
    //std::cerr<<duration<<" seconds"<<std::endl;
    std::stringstream ss;
    for(size_t i = 0; i < result.size(); ++i)
    {
        if(i != 0)
        ss << " ";
        ss << result[i].first+result[i].second;
    }
    std::string lac_result = ss.str();
    
    return PyBytes_FromString(lac_result.data());
}
static PyMethodDef ThulacMethods[] = {
    {"thulac", method_thulac, METH_VARARGS, "Python interface for segmentaion C++ library function"},
    {NULL, NULL, 0, NULL}
};


static struct PyModuleDef thulacmodule = {
    PyModuleDef_HEAD_INIT,
    "thulac",
    "Python interface for segmentaion C++ library function",
    -1,
    ThulacMethods
};
PyMODINIT_FUNC PyInit_thulac(void) {
    return PyModule_Create(&thulacmodule);
}

