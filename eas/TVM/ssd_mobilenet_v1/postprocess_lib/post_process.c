
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <math.h>
#include "ssd_mobilenet_v1.h"
#include <sys/time.h>

#define BOX_PREDICTOR_NUM 6
#define BBoxNum 6

float *generate_Sk(int m, float s_min, float s_max)
{
	float *Sk = (float *)malloc(sizeof(float)*(m+1));
	if(Sk==NULL)
	{
		printf("malloc Sk error\n");
		return NULL;
	}

	int x;
	float s_sub = s_max - s_min;
	for(x = 0; x < m; x++)
	{
		Sk[x] = s_min + s_sub / (m - 1) * x;	
	}

	return Sk;
	
}

void generate_box_ratio(float box_ratio[BOX_PREDICTOR_NUM][BBoxNum], float ratio[BBoxNum])
{
	int x,y;
	for(y = 0; y < BOX_PREDICTOR_NUM; y++)
	{
		//printf("BBox_ratio[%d]=[ ", y);
		int bboxnum =  BBoxNum;
		if(y == 0)
			bboxnum = 3;
		for(x = 0; x < bboxnum; x++)
		{
			box_ratio[y][x] = ratio[x];
		//	printf("%1.3f ", box_ratio[y][x]);	
		}
		//printf("]\n");	
	}
}

void generate_box_sk(float box_sk[BOX_PREDICTOR_NUM][BBoxNum], float *Sk)
{
	int x,y;
	for(y = 0; y < BOX_PREDICTOR_NUM; y++)
	{
		//printf("BBox_sk[%d]=[ ", y);
		int bboxnum =  BBoxNum;
		if(y == 0)
		{	
			bboxnum = 3;
			box_sk[y][0] = 0.1;
			box_sk[y][1] = 0.2;
			box_sk[y][2] = 0.2;			
			//for(x = 0; x < bboxnum; x++)
			//{
			//	printf("%1.3f ", box_sk[y][x]);	
			//}			
		}else
		{
			for(x = 0; x < bboxnum; x++)
			{
				float tmp_sk = Sk[y];
				if(x == (bboxnum - 1))
					tmp_sk = sqrt(Sk[y]*Sk[y+1]);
				box_sk[y][x] = tmp_sk;
				//printf("%1.3f ", box_sk[y][x]);	
			}
		}
		//printf("]\n");	
	}
}

void generate_bbox(float *box_ratio, float *box_sk, int bbox_num, float width, float height, float *bbox)
{
	static float dw[BBoxNum];
	static float dh[BBoxNum];

	int x,y,z;
	float dcx,dcy;
	for(x = 0;x < bbox_num;x++)
	{
		dw[x] = box_sk[x] * sqrt(box_ratio[x]);
		dh[x] = box_sk[x] / sqrt(box_ratio[x]);
	}
	
	for(y = 0;y < height; y++)
		for(z = 0;z < width; z++)
		{
			dcy = (y + 0.5) / height;
			dcx = (z + 0.5) / width;
			for(x = 0;x < bbox_num; x++)
			{	
				int idx = y*width*bbox_num*4 + z*bbox_num*4 + x*4; 
				bbox[idx    ] = dcy - dh[x]/2;
				bbox[idx + 1] = dcx - dw[x]/2;
				bbox[idx + 2] = dcy + dh[x]/2;
				bbox[idx + 3] = dcx + dw[x]/2;
			}
		}

}

void generate_bbox_pp(float *output_bbox, float *input_bbox_encode, float *box_ratio, float *box_sk, int bbox_num, float width, float height)
{
	static float dw[BBoxNum];
	static float dh[BBoxNum];

	int x,y,z;
	float dcx,dcy;
	float pcx, pcy, ph, pw;

	for(x = 0;x < bbox_num;x++)
	{
		dw[x] = box_sk[x] * sqrt(box_ratio[x]);
		dh[x] = box_sk[x] / sqrt(box_ratio[x]);
	}
	
	for(y = 0;y < height; y++)
		for(z = 0;z < width; z++)
		{
			dcy = (y + 0.5) / height;
			dcx = (z + 0.5) / width;
			for(x = 0;x < bbox_num; x++)
			{	
				int idx = y*width*bbox_num*4 + z*bbox_num*4 + x*4;
 				pcy = input_bbox_encode[idx    ] / 10.0 * dh[x] + dcy;
				pcx = input_bbox_encode[idx + 1] / 10.0 * dw[x] + dcx;
				ph  = exp(input_bbox_encode[idx + 2] / 5.0) * dh[x];
				pw  = exp(input_bbox_encode[idx + 3] / 5.0) * dw[x];

				output_bbox[idx    ] = pcy - ph/2;
				output_bbox[idx + 1] = pcx - pw/2;
				output_bbox[idx + 2] = pcy + ph/2;
				output_bbox[idx + 3] = pcx + pw/2;
			}
		}

}

float sigmoid(float x)//f(z) = 1 / (1 + exp( − z))
{
	return (1.0 / (1 + exp(-x)));
}

void sigmoid_class_pred(float *class_pred, int data_num)
{
	int x;
	for(x = 0; x < data_num; x++)
	{
		class_pred[x] = sigmoid(class_pred[x]);
	//	printf("[%d] = %2.6lf\n", x, class_pred[x]);
	}
}

//post_process(im.data, im.w, im.h, loc_output_buf, cls_output_buf);
void post_process(float *im_data, int im_w, int im_h, float *loc_output_buf, float *cls_output_buf)
{
	double time1, time2;
	int x;
	image im;
	im.data = im_data;
	im.w = im_w;
	im.h = im_h;
	im.c = 3;
	
	char labels[91][256];
	char line[256];
	FILE *fp3;
        if( (fp3 = fopen("coco_labels91.txt", "r")) == NULL)
		printf("CANNOT OPEN\n");
        for( x = 0; x < 91; x++)//coco 91 class labels
	{
		fgets(line, 256 ,fp3);
	        memcpy(labels[x], line, 256);
		labels[x][strlen(labels[x])-1]= '\0';
        }
        fclose(fp3);

	const int bp_iofm_hw_set[BOX_PREDICTOR_NUM]   = {  19,  10,   5,   3,   2,   1};
	const float s_min = 0.2, s_max = 0.95;
	float ratio[BBoxNum] = {1.0, 2.0, 0.5, 3.0, 0.3333, 1.0};
	const int bbox_num_set[BBoxNum] = {3, 6, 6, 6, 6, 6};

	float *Sk = generate_Sk( BOX_PREDICTOR_NUM, s_min, s_max);
	Sk[BOX_PREDICTOR_NUM] = 1.0;

	float box_ratio[BOX_PREDICTOR_NUM][BBoxNum];
	float box_sk[BOX_PREDICTOR_NUM][BBoxNum];

	generate_box_ratio( box_ratio, ratio);
	generate_box_sk(box_sk, Sk);

	static float bbox_buf[1917*4];

	int bbox_offset;
	for(x = 0, bbox_offset = 0; x < BOX_PREDICTOR_NUM; x++)
	{
		generate_bbox_pp(bbox_buf + bbox_offset, loc_output_buf + bbox_offset, box_ratio[x], box_sk[x], bbox_num_set[x], bp_iofm_hw_set[x], bp_iofm_hw_set[x]);
		bbox_offset += bbox_num_set[x]*bp_iofm_hw_set[x]*bp_iofm_hw_set[x]*4;
	}
	sigmoid_class_pred(cls_output_buf, 1917*91);

        int nboxes = 0;
    	float nms=.6;
	float thresh = 0.3;
	detection *dets = get_network_boxes(cls_output_buf, bbox_buf, im.w, im.h, thresh, 1, &nboxes);
        //printf("%d\n", nboxes);
	
	//int new_nboxes = 0;
	for(x=0;x<nboxes;x++)
	{
		if(dets[x].objectness!=0.0)
		{
			printf("[%3d]:h=%f,w=%f,x=%f,y=%f,objectness=%f\n",x,dets[x].bbox.h,dets[x].bbox.w,dets[x].bbox.x,dets[x].bbox.y,dets[x].objectness);
			//new_nboxes++;
		}
	}
        if (nms) do_nms_sort(dets, nboxes, 90, nms);

	draw_detections(im, dets, nboxes, thresh, labels + 1, NULL, 90);

	free_detections(dets, nboxes);
	free(Sk);
}


