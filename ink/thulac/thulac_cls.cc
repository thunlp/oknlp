#include "thulac.h"
#include <sstream>
#include <fstream>
#include <Python.h>
#include <structmember.h>
using std::cin;
using std::cout;
using std::endl;
typedef struct {
    PyObject_HEAD // no semicolon
    THULAC *lac;
} THUlac;

static void THUlac_dealloc(THUlac *self) {
    delete self->lac ;
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *THUlac_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    THUlac *self;
    self = (THUlac *)type->tp_alloc(type, 0);
    self->lac = NULL;
    return (PyObject *)self;
}

static PyObject* THUlac_init(THUlac *self, PyObject *args, PyObject *kwds) {
    //char* raw = NULL;
    char* model_path = NULL;
    char* user_specified_dict_name=NULL;
    bool useT2S = false;
    bool seg_only = true;
    bool useFilter = false;
    /* Parse arguments */
    if(!PyArg_ParseTuple(args, "s", &model_path)){
        return NULL;
    };
    self->lac = new THULAC();
    self->lac->init(model_path, user_specified_dict_name, seg_only, useT2S, useFilter);
     return 0;
}

static PyObject* THUlac_cws(THUlac *self, PyObject *args, PyObject *kwds) {
    char* raw = NULL; 
    PyArg_ParseTuple(args, "s", &raw);
    THULAC_result result;
    std::stringstream ss;
    for(size_t i = 0; i < result.size(); ++i)
    {
        if(i != 0)
        ss << " ";
        ss << result[i].first+result[i].second;
    }
    std::string lac_result = ss.str();
    
    return PyBytes_FromString(lac_result.data());
};

static PyMemberDef THUlac_members[] = {
 { "THUlac", T_PYSSIZET, offsetof(THUlac, lac) , READONLY, "The object of thulac" },
 {NULL, 0, 0, 0, NULL}
};

static PyMethodDef THUlac_methods[] = {
    {"THUlac_cws", (PyCFunction)THUlac_cws, METH_VARARGS,
     "Interface for segmentation"
    },
    {NULL}  /* Sentinel */
};

static PyTypeObject THUlacType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "THUlac",             /* tp_name */
    sizeof(THUlac),             /* tp_basicsize */
    0,                         /* tp_itemsize */
    (destructor)THUlac_dealloc, /* tp_dealloc */
    0,                         /* tp_print */
    0,                         /* tp_getattr */
    0,                         /* tp_setattr */
    0,                         /* tp_reserved */
    0,                         /* tp_repr */
    0,                         /* tp_as_number */
    0,                         /* tp_as_sequence */
    0,                         /* tp_as_mapping */
    0,                         /* tp_hash  */
    0,                         /* tp_call */
    0,                         /* tp_str */
    0,                         /* tp_getattro */
    0,                         /* tp_setattro */
    0,                         /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT |
        Py_TPFLAGS_BASETYPE,   /* tp_flags */
    "THUlac objects",           /* tp_doc */
    0,                         /* tp_traverse */
    0,                         /* tp_clear */
    0,                         /* tp_richcompare */
    0,                         /* tp_weaklistoffset */
    0,                         /* tp_iter */
    0,                         /* tp_iternext */
    THUlac_methods,             /* tp_methods */
    THUlac_members,             /* tp_members */
    0,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)THUlac_init,      /* tp_init */
    0,                         /* tp_alloc */
    THUlac_new,                 /* tp_new */
};



// static PyMethodDef ThulacMethods[] = {
//     {"THUlac", method_thulac, METH_VARARGS, "Python interface for segmentaion C++ library function"},
//     {NULL, NULL, 0, NULL}
// };




static struct PyModuleDef lacthumodule = {
    PyModuleDef_HEAD_INIT,
    "THUlac",
    "Python interface for segmentaion C++ library function",
    -1,
    NULL, NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC PyInit_lacthu(void)
{
    PyObject* m;

    if (PyType_Ready(&THUlacType) < 0)
        return NULL;

    m = PyModule_Create(&lacthumodule);
    if (m == NULL)
        return NULL;

    Py_INCREF(&THUlacType);
    PyModule_AddObject(m, "THUlac", (PyObject *)&THUlacType);
    return m;
}


