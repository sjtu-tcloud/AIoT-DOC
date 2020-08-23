//
//  pictureRecognition.cpp
//  MNN
//
//  Created by MNN on 2018/05/14.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include <MNN/ImageProcess.hpp>
#include <MNN/Interpreter.hpp>
#define MNN_OPEN_TIME_TRACE
#include <algorithm>
#include <fstream>
#include <functional>
#include <memory>
#include <sstream>
#include <vector>
#include <MNN/AutoTime.hpp>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"
#include <sys/time.h>

double what_time_is_it_now()
{
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

using namespace MNN;
using namespace MNN::CV;

int main(int argc, const char* argv[]) {
    if (argc < 3) {
        MNN_PRINT("Usage: ./pictureTest.out model.mnn input.jpg [word.txt]\n");
        return 0;
    }
    std::shared_ptr<Interpreter> net(Interpreter::createFromFile(argv[1]));
    ScheduleConfig config;
    config.type  = MNN_FORWARD_AUTO;
    // BackendConfig bnconfig;
    // bnconfig.precision = BackendConfig::Precision_Low;
    // config.backendConfig = &bnconfig;
    auto session = net->createSession(config);

    auto input = net->getSessionInput(session, "input_tensor");
    auto shape = input->shape();
    shape[0]   = 1;
    net->resizeTensor(input, shape);
    net->resizeSession(session);

	static unsigned char labels[1000][256];
	char line[256];
	FILE *fp3;
	if( (fp3 = fopen(argv[3], "r")) == NULL)
		printf("CANNOT OPEN\n");
	int x;
	for( x = 0; x < 1000; x++)
	{
		fgets(line, 256 ,fp3);
		memcpy(labels[x], line, 256);
		//printf("[%d]:%s\n", x, labels[x]);
	}
	fclose(fp3);

    {
 //       auto dims    = input->shape();
        int size_w   = 0;
        int size_h   = 0;
        int bpp      = 0;
        bpp          = input->channel();
        size_h       = input->height();
        size_w       = input->width();
        if (bpp == 0)
            bpp = 1;
        if (size_h == 0)
            size_h = 1;
        if (size_w == 0)
            size_w = 1;
        MNN_PRINT("input: w:%d , h:%d, bpp: %d\n", size_w, size_h, bpp);

        auto inputPatch = argv[2];
        int width, height, channel;
        auto inputImage = stbi_load(inputPatch, &width, &height, &channel, 4);
        if (nullptr == inputImage) {
            MNN_ERROR("Can't open %s\n", inputPatch);
            return 0;
        }
        MNN_PRINT("origin size: %d, %d, %d\n", width, height, channel);
        Matrix trans;
        trans.setScale((float)(width-1) / (size_w-1), (float)(height-1) / (size_h-1));

        ImageProcess::Config config;
        config.filterType = BILINEAR;
   //     float mean[3]     = {103.94f, 116.78f, 123.68f};
	float mean[3]     = {123.68f, 116.78f, 103.94f};
        float normals[3] = {1.0f, 1.0f, 1.0f};

        ::memcpy(config.mean, mean, sizeof(mean));
        ::memcpy(config.normal, normals, sizeof(normals));
        config.sourceFormat = RGBA;
        config.destFormat   = RGB;

        std::shared_ptr<ImageProcess> pretreat(ImageProcess::create(config));
        pretreat->setMatrix(trans);

//add s
//	auto nhwcTensor = new Tensor(input, Tensor::CAFFE_C4);
//	pretreat->convert((uint8_t*)inputImage, width, height, 0, nhwcTensor);
//	input->copyFromHostTensor(nhwcTensor);
//add e
        pretreat->convert((uint8_t*)inputImage, width, height, 0, input);

        stbi_image_free(inputImage);
    }

	double time1, time2;
	time1 = what_time_is_it_now();
    	net->runSession(session);
	time2 = what_time_is_it_now();
	printf("Mobilenet v1: %5.8lf s\n\n", time2 - time1);

auto output = net->getSessionOutput(session, "softmax_tensor");

 {
        auto dimType = output->getDimensionType();
        if (output->getType().code != halide_type_float) {
            dimType = Tensor::TENSORFLOW;
        }
        std::shared_ptr<Tensor> outputUser(new Tensor(output, dimType));
        MNN_PRINT("output size:%d\n", outputUser->elementSize());
        output->copyToHostTensor(outputUser.get());
        auto type = outputUser->getType();

        auto size = outputUser->elementSize();
        std::vector<std::pair<int, float>> tempValues(size);
        if (type.code == halide_type_float) {
            auto values = outputUser->host<float>();
            for (int i = 0; i < size; ++i) {
                tempValues[i] = std::make_pair(i, values[i]);
            }
        }
        if (type.code == halide_type_uint && type.bytes() == 1) {
            auto values = outputUser->host<uint8_t>();
            for (int i = 0; i < size; ++i) {
                tempValues[i] = std::make_pair(i, values[i]);
            }
        }
        // Find Max
        std::sort(tempValues.begin(), tempValues.end(),
                  [](std::pair<int, float> a, std::pair<int, float> b) { return a.second > b.second; });

        int length = size > 5 ? 5 : size;
	    for (int i = 0; i < length; ++i) {
		MNN_PRINT("[%d]%s: %f\n", tempValues[i].first, labels[tempValues[i].first-1], tempValues[i].second);
	    }

}

    return 0;
}
