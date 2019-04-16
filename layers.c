#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// Include SSE intrinsics
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <immintrin.h>
#include <x86intrin.h>
#endif

// Include OpenMP
#include <omp.h>

#include "layers.h"
#include "volume.h"

conv_layer_t *make_conv_layer(int input_width, int input_height, int input_depth, int filter_width, int num_filters,
        int stride, int pad) {
    conv_layer_t *l = (conv_layer_t *) malloc(sizeof(conv_layer_t));

    l->output_depth = num_filters;
    l->filter_width = filter_width;
    l->input_depth = input_depth;
    l->input_width = input_width;
    l->input_height = input_height;

    l->filter_height = l->filter_width;
    l->stride = stride;
    l->pad = pad;

    l->output_width = (l->input_width + l->pad * 2 - l->filter_width) /
        l->stride + 1;
    l->output_height = (l->input_height + l->pad * 2 - l->filter_height) /
        l->stride + 1;

    l->filters = malloc(sizeof(volume_t *) * num_filters);
    for (int i = 0; i < num_filters; i++) {
        l->filters[i] = make_volume(l->filter_width, l->filter_height,
            l->input_depth, 0.0);
    }

    l->bias = 0.0;
    l->biases = make_volume(1, 1, l->output_depth, l->bias);

    return l;
}

// Performs the forward pass for a convolutional layer by convolving each one
// of the filters with a particular input, and placing the result in the output
// array.
//
// One way to think about convolution in this case is that we have one of the
// layer's filters (a 3D array) that is superimposed on one of the layer's
// inputs (a second 3D array) that has been implicitly padded with zeros. Since
// convolution is a sum of products (described below), we don't actually have
// to add any zeros to the input volume since those terms will not contribute
// to the convolution. Instead, for each position in the filter, we just make
// sure that we are in bounds for the input volume.
//
// Essentially, the filter is "sliding" across the input, in both the x and y
// directions, where we increment our position in each direction by using the
// stride parameter.
//
// At each position, we compute the sum of the elementwise product of the filter
// and the part of the array it's covering. For instance, let's consider a 2D
// case, where the filter (on the left) is superimposed on some part of the
// input (on the right).
//
//   Filter             Input
//  -1  0  1           1  2  3
//  -1  0  1           4  5  6
//  -1  0  1           7  8  9
//
// Here, the sum of the elementwise product is:
//    Filter[0][0] * Input[0][0] + Filter[0][1] * Input[0][1] + ...
//    = -1 * 1 + 0 * 2 + ... + 0 * 8 + 1 * 9
//    = 6
//
// The 3D case is essentially the same, we just have to sum over the other
// dimension as well. Also, since volumes are internally represented as 1D
// arrays, we must use the volume_get and volume_set commands to access elements
// at a coordinate (x, y, d). Finally, we add the corresponding bias for the
// filter to the sum before putting it into the output volume.
void conv_forward(conv_layer_t *l, volume_t **inputs, volume_t **outputs, int start, int end) {
	int stride = l->stride;
	int depth = l->output_depth;
	int height = l->output_height;
	int width = l->output_width;
	int pad = l->pad;
	double resultarr[4];
	__m256d sumvect = _mm256_setzero_pd();
	__m256d invect = _mm256_setzero_pd();
	__m256d filvect = _mm256_setzero_pd();
	double* weightarr = l->biases->weights;
	for (int i = start; i <= end; i++) {
        volume_t *in = inputs[i];
        volume_t *out = outputs[i];
		int inheight = in->height;
		int inwidth = in->width;
		int indepth = in->depth;
		double* inweights = in->weights;
        for(int f = 0; f < depth; f++) {
            volume_t *filter = l->filters[f];
			int filheight = filter->height;
			int filwidth = filter->width;
			int fildepth = filter->depth;
			double* filweights = filter->weights;
			int y = -pad;
            for(int out_y = 0; out_y < height; y += stride, out_y++) {
				int x = -pad;
                for(int out_x = 0; out_x < width; x += stride, out_x++) {

                    // Take sum of element-wise product
					double sum = 0.0;
					sumvect = _mm256_setzero_pd();
					for (int fy = 0; fy < filheight; fy++) {
						int in_y = y + fy;
						for (int fx = 0; fx < filwidth; fx++) {
							int in_x = x + fx;
							if (in_y >= 0 && in_y < inheight && in_x >= 0 && in_x < inwidth) {
								if (fildepth == 3) {
									sum += filweights[((filwidth * fy) + fx) * 3] * inweights[((inwidth * in_y) + in_x) * indepth];
									sum += filweights[((filwidth * fy) + fx) * 3 + 1] * inweights[((inwidth * in_y) + in_x) * indepth + 1];
									sum += filweights[((filwidth * fy) + fx) * 3 + 2] * inweights[((inwidth * in_y) + in_x) * indepth + 2];
								}
								else if (fildepth == 16) {
									for (int fd = 0; fd < fildepth; fd++) {
										sum += volume_get(filter, fx, fy, fd) * volume_get(in, in_x, in_y, fd);
									}
									/*invect = _mm256_loadu_pd((__m256d*)(inweights + ((inwidth * in_y) + in_x) * indepth));
									filvect = _mm256_loadu_pd((__m256d*)(filweights + ((filwidth * fy) + fx) * 16));
									sumvect = _mm256_add_pd(_mm256_mul_pd(invect, filvect), sumvect);
									invect = _mm256_loadu_pd((__m256d*)(inweights + ((inwidth * in_y) + in_x) * indepth + 4));
									filvect = _mm256_loadu_pd((__m256d*)(filweights + ((filwidth * fy) + fx) * 16 + 4));
									sumvect = _mm256_add_pd(_mm256_mul_pd(invect, filvect), sumvect);
									invect = _mm256_loadu_pd((__m256d*)(inweights + ((inwidth * in_y) + in_x) * indepth + 8));
									filvect = _mm256_loadu_pd((__m256d*)(filweights + ((filwidth * fy) + fx) * 16 + 8));
									sumvect = _mm256_add_pd(_mm256_mul_pd(invect, filvect), sumvect);
									invect = _mm256_loadu_pd((__m256d*)(inweights + ((inwidth * in_y) + in_x) * indepth + 12));
									filvect = _mm256_loadu_pd((__m256d*)(filweights + ((filwidth * fy) + fx) * 16 + 12));
									sumvect = _mm256_add_pd(_mm256_mul_pd(invect, filvect), sumvect);
									_mm256_storeu_pd(resultarr, sumvect);
									sum += resultarr[0] + resultarr[1] + resultarr[2] + resultarr[3];*/
								}
								else if (fildepth == 20) {
									for (int fd = 0; fd < fildepth; fd++) {
										sum += volume_get(filter, fx, fy, fd) * volume_get(in, in_x, in_y, fd);
									}
									/*
									invect = _mm256_loadu_pd((__m256d*)(inweights + ((inwidth * in_y) + in_x) * indepth));
									filvect = _mm256_loadu_pd((__m256d*)(filweights + ((filwidth * fy) + fx) * 20));
									sumvect = _mm256_add_pd(_mm256_mul_pd(invect, filvect), sumvect);
									invect = _mm256_loadu_pd((__m256d*)(inweights + ((inwidth * in_y) + in_x) * indepth + 4));
									filvect = _mm256_loadu_pd((__m256d*)(filweights + ((filwidth * fy) + fx) * 20 + 4));
									sumvect = _mm256_add_pd(_mm256_mul_pd(invect, filvect), sumvect);
									invect = _mm256_loadu_pd((__m256d*)(inweights + ((inwidth * in_y) + in_x) * indepth + 8));
									filvect = _mm256_loadu_pd((__m256d*)(filweights + ((filwidth * fy) + fx) * 20 + 8));
									sumvect = _mm256_add_pd(_mm256_mul_pd(invect, filvect), sumvect);
									invect = _mm256_loadu_pd((__m256d*)(inweights + ((inwidth * in_y) + in_x) * indepth + 12));
									filvect = _mm256_loadu_pd((__m256d*)(filweights + ((filwidth * fy) + fx) * 20 + 12));
									sumvect = _mm256_add_pd(_mm256_mul_pd(invect, filvect), sumvect);
									invect = _mm256_loadu_pd((__m256d*)(inweights + ((inwidth * in_y) + in_x) * indepth + 16));
									filvect = _mm256_loadu_pd((__m256d*)(filweights + ((filwidth * fy) + fx) * 20 + 16));
									sumvect = _mm256_add_pd(_mm256_mul_pd(invect, filvect), sumvect);
									_mm256_storeu_pd(resultarr, sumvect);
									sum += resultarr[0] + resultarr[1] + resultarr[2] + resultarr[3];*/
								}

							}
						}
					}
                    sum += weightarr[f];
                    volume_set(out, out_x, out_y, f, sum);
                }
            }
        }
    }
}

void conv_load(conv_layer_t *l, const char *file_name) {
    int filter_width, filter_height, depth, filters;

    FILE *fin = fopen(file_name, "r");

    fscanf(fin, "%d %d %d %d", &filter_width, &filter_height, &depth, &filters);
    assert(filter_width == l->filter_width);
    assert(filter_height == l->filter_height);
    assert(depth == l->input_depth);
    assert(filters == l->output_depth);

    for(int f = 0; f < filters; f++) {
        for (int x = 0; x < filter_width; x++) {
            for (int y = 0; y < filter_height; y++) {
                for (int d = 0; d < depth; d++) {
                    double val;
                    fscanf(fin, "%lf", &val);
                    volume_set(l->filters[f], x, y, d, val);
                }
            }
        }
    }

    for(int d = 0; d < l->output_depth; d++) {
        double val;
        fscanf(fin, "%lf", &val);
        volume_set(l->biases, 0, 0, d, val);
    }

    fclose(fin);
}

relu_layer_t *make_relu_layer(int input_width, int input_height, int input_depth) {
    relu_layer_t *l = (relu_layer_t *) malloc(sizeof(relu_layer_t));

    l->input_depth = input_depth;
    l->input_width = input_width;
    l->input_height = input_height;

    l->output_width = l->input_width;
    l->output_height = l->input_height;
    l->output_depth = l->input_depth;

    return l;
}

// Applies the Rectifier Linear Unit (ReLU) function to the input, which sets
// output(x, y, d) to max(0.0, input(x, y, d)).
void relu_forward(relu_layer_t *l, volume_t **inputs, volume_t **outputs, int start, int end) {
	int depth = l->output_depth;
	int height = l->output_height;
	int width = l->output_width;
    for (int i = start; i <= end; i++) {
        for (int x = 0; x < width; x++) {
            for (int y = 0; y < height; y++) {
                for (int d = 0; d < depth; d++) {
					double vol = volume_get(inputs[i], x, y, d);
                    double value = (vol < 0.0) ? 0.0 : vol;
                    volume_set(outputs[i], x, y, d, value);
                }
            }
        }
    }
}

pool_layer_t *make_pool_layer(int input_width, int input_height, int input_depth, int pool_width, int stride) {
    pool_layer_t *l = (pool_layer_t *) malloc(sizeof(pool_layer_t));

    l->pool_width = pool_width;
    l->input_depth = input_depth;
    l->input_width = input_width;
    l->input_height = input_height;

    l->pool_height = l->pool_width;
    l->stride = stride;
    l->pad = 0;

    l->output_depth = input_depth;
    l->output_width = floor((l->input_width + l->pad * 2 - l->pool_width) / l->stride + 1);
    l->output_height = floor((l->input_height + l->pad * 2 - l->pool_height) / l->stride + 1);

    return l;
}

// This is like the convolutional layer in that we are sliding across the input
// volume, but instead of having a filter that we use to find the sum of an
// elementwise product, we instead just output the max value of some part of
// the image. For instance, if we consider a 2D case where the following is the
// part of the input that we are considering:
//
//     1 3 5
//     4 2 1
//     2 2 2
//
// then the value of the corresponding element in the output is 5 (since that
// is the maximum element). This effectively compresses the input.
void pool_forward(pool_layer_t *l, volume_t **inputs, volume_t **outputs, int start, int end) {
	int stride = l->stride;
	int depth = l->output_depth;
	int height = l->output_height;
	int width = l->output_width;
	int poolwidth = l->pool_width;
	int poolheight = l->pool_height;
	int pad = l->pad;
    for (int i = start; i <= end; i++) {
        volume_t *in = inputs[i];
        volume_t *out = outputs[i];
		int inheight = in->height;
		int inwidth = in->width;
        int n = 0;
        for(int d = 0; d < depth; d++) {
            int x = -pad;
            for(int out_x = 0; out_x < width; x += stride, out_x++) {
                int y = -pad;
                for(int out_y = 0; out_y < height; y += stride, out_y++) {

                    double max = -INFINITY;
                    for(int fx = 0; fx < poolwidth; fx++) {
                        for(int fy = 0; fy < poolheight; fy++) {
                            int in_y = y + fy;
                            int in_x = x + fx;
                            if(in_x >= 0 && in_x < inwidth && in_y >= 0 && in_y < inheight) {
                                double v = volume_get(in, in_x, in_y, d);
                                if(v > max) {
                                    max = v;
                                }
                            }
                        }
                    }

                    n++;
                    volume_set(out, out_x, out_y, d, max);
                }
            }
        }
    }
}

fc_layer_t *make_fc_layer(int input_width, int input_height, int input_depth, int num_neurons) {
    fc_layer_t *l = (fc_layer_t *) malloc(sizeof(fc_layer_t));

    l->output_depth = num_neurons;
    l->input_depth = input_depth;
    l->input_width = input_width;
    l->input_height = input_height;

    l->num_inputs = l->input_width * l->input_height * l->input_depth;
    l->output_width = 1;
    l->output_height = 1;

    l->filters = (volume_t **) malloc(sizeof(volume_t *) * num_neurons);
    for (int i = 0; i < l->output_depth; i++) {
        l->filters[i] = make_volume(1, 1, l->num_inputs, 0.0);
    }

    l->bias = 0.0;
    l->biases = make_volume(1, 1, l->output_depth, l->bias);

    return l;
}

// Computes the dot product (i.e. the sum of the elementwise product) of the
// input's weights with each of the filters. Note that these filters are not
// the same as the filters for the convolutional layer.
void fc_forward(fc_layer_t *l, volume_t **inputs, volume_t **outputs, int start, int end) {
    for (int j = start; j <= end; j++) {
        volume_t *in = inputs[j];
        volume_t *out = outputs[j];

        for(int i = 0; i < l->output_depth;i++) {
            double dot = 0.0;
            for(int d = 0; d < l->num_inputs; d++) {
                dot += in->weights[d] * l->filters[i]->weights[d];
            }
            dot += l->biases->weights[i];
            out->weights[i] = dot;
        }
    }
}

void fc_load(fc_layer_t *l, const char *filename) {
    FILE *fin = fopen(filename, "r");

    int num_inputs;
    int output_depth;
    fscanf(fin, "%d %d", &num_inputs, &output_depth);
    assert(output_depth == l->output_depth);
    assert(num_inputs == l->num_inputs);

    for(int i = 0; i < l->output_depth; i++)
        for(int j = 0; j < l->num_inputs; j++) {
            fscanf(fin, "%lf", &(l->filters[i]->weights[j]));
        }

    for(int i = 0; i < l->output_depth; i++) {
        fscanf(fin, "%lf", &(l->biases->weights[i]));
    }

    fclose(fin);
}

softmax_layer_t *make_softmax_layer(int input_width, int input_height, int input_depth) {
    softmax_layer_t *l = (softmax_layer_t*) malloc(sizeof(softmax_layer_t));

    l->input_depth = input_depth;
    l->input_width = input_width;
    l->input_height = input_height;

    l->output_width = 1;
    l->output_height = 1;
    l->output_depth = l->input_width * l->input_height * l->input_depth;

    l->likelihoods = (double*) malloc(sizeof(double) * l->output_depth);

    return l;
}

// This function converts an input's weights array into a probability
// distribution by using the following formula:
//
// likelihood[i] = exp(in->weights[i]) / sum(exp(in->weights))
//
// To increase the numerical stability of taking the exponential of a value, we
// subtract the maximum input weights from each weight before taking the
// exponential. This yields exactly the same results as the expression above,
// but is more resilient to floating point errors.
void softmax_forward(softmax_layer_t *l, volume_t **inputs, volume_t **outputs, int start, int end) {
    double likelihoods[l->output_depth];

    for (int j = start; j <= end; j++) {
        volume_t *in = inputs[j];
        volume_t *out = outputs[j];

        // Compute max activation (used to compute exponentials)
        double amax = in->weights[0];
        for(int i = 1; i < l->output_depth; i++) {
            if (in->weights[i] > amax) {
                amax = in->weights[i];
            }
        }

        // Compute exponentials in a numerically stable way
        double total = 0.0;
        for(int i = 0; i < l->output_depth; i++) {
            double e = exp(in->weights[i] - amax);
            total += e;
            likelihoods[i] = e;
        }

        // Normalize and output to sum to one
        for(int i = 0; i < l->output_depth; i++) {
            out->weights[i] = likelihoods[i] / total;
        }
    }
}
