/* create a Python extension to compute connectivity for a multii-leel image*/
#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdio.h>
#include <stdlib.h>

#define IM(offset) (image[offset])
#define LI(offset) (labelimage[offset])
#define LIR(offset) (lmap[labelimage[offset]])

#define UNKNOWN 0

static unsigned int *lmap;
static unsigned int *point;
static unsigned int *label;
static unsigned int *color
static unsigned int *parent

static inline merge(int a, int b) {
    if (a < b) {
        _ = a; a = b; b = _;
    }
}   

// Function to compute connectivity
static PyObject *connectivity(PyObject *self, PyObject *args) {
    PyArrayObject *input;


    // Parse the input tuple
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &input)) {
        return NULL;
    }

    // get pointer to the data
    lmap = (unsigned int *)PyArray_DATA(input);
    // get width of input image
    unsigned int width = PyArray_DIM(input, 1);
    // get height of input image
    unsigned int height = PyArray_DIM(input, 0);

    // create a new array to store the label image
    npy_intp dims[2] = {height, width};
    PyArrayObject *labelimage = (PyArrayObject *)PyArray_SimpleNew(2, dims, NPY_UINT32);
    unsigned int *label = (unsigned int *)PyArray_DATA(labelimage);
    

    N = -height;
    NW = -height - 1;
    NE = -height + 1;
    W = -1;
    C = 0;
    maxlabel = 0;
    buflen = 10000;
    point = calloc(buflen, sizeof(int));
    color = calloc(buflen, sizeof(int));
    parent = calloc(buflen, sizeof(int));

    for (row=0; row<height; row++) {
        for (col=0; col<width; col++) {

            curlabel = maxint()
            if (col == 0) {
                if (IM[C] == IM[N]))
                    curlabel = LIR(N)
            }
            else {
                if (IM[C] == IM[W]))
                    curlabel = LIR(W);
                else if (IM[C] == IM[N]))
                    curlabel = LIR(N);
            }
            if (curlabel == UNKNOWN) {
                curlabel = maxlabel + 1;
                if (maxlabel >= buflen) {
                    // need to extend the buffers
                    buflen += 10000;
                    point = realloc(point, buflen*sizeof(int));
                    color = realloc(color, buflen*sizeof(int));
                    parent = realloc(parent, buflen*sizeof(int));
                }
                point[curlabel] = row * height + col;  // index of initial point
                color[curlabel] = IM[C];
                parent[curlabel] = curlabel;    // parent of a node is itself
                maxlabel += 1;
            }
            offset = row*width + col;
            if (LI(offset) != 0) {
                if (row > 0) {
                    if (col > 0) {
                        if (LI(offset + NW) != 0) {
                            LIR(offset) = LIR(offset + NW);
                        } else {
                            LIR(offset) = LI(offset);
                        }
                    } else {
                        LIR(offset) = LI(offset);
                    }
                } else {
                    LIR(offset) = LI(offset);
                }
            }
        }
    }


