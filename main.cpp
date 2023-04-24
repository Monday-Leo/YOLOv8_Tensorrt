#include <opencv2/opencv.hpp>
#include "cpm.hpp"
#include "infer.hpp"
#include "yolo.hpp"

using namespace std;

static const char *cocolabels[] = { "person",        "bicycle",      "car",
								   "motorcycle",    "airplane",     "bus",
								   "train",         "truck",        "boat",
								   "traffic light", "fire hydrant", "stop sign",
								   "parking meter", "bench",        "bird",
								   "cat",           "dog",          "horse",
								   "sheep",         "cow",          "elephant",
								   "bear",          "zebra",        "giraffe",
								   "backpack",      "umbrella",     "handbag",
								   "tie",           "suitcase",     "frisbee",
								   "skis",          "snowboard",    "sports ball",
								   "kite",          "baseball bat", "baseball glove",
								   "skateboard",    "surfboard",    "tennis racket",
								   "bottle",        "wine glass",   "cup",
								   "fork",          "knife",        "spoon",
								   "bowl",          "banana",       "apple",
								   "sandwich",      "orange",       "broccoli",
								   "carrot",        "hot dog",      "pizza",
								   "donut",         "cake",         "chair",
								   "couch",         "potted plant", "bed",
								   "dining table",  "toilet",       "tv",
								   "laptop",        "mouse",        "remote",
								   "keyboard",      "cell phone",   "microwave",
								   "oven",          "toaster",      "sink",
								   "refrigerator",  "book",         "clock",
								   "vase",          "scissors",     "teddy bear",
								   "hair drier",    "toothbrush" };


yolo::Image cvimg(const cv::Mat &image) { return yolo::Image(image.data, image.cols, image.rows); }

void single_inference(std::string img_path, std::string model_path) {
	cv::Mat image = cv::imread(img_path);

	float confidence_threshold = 0.25f;
	float nms_threshold = 0.5f;
	auto yolo = yolo::load(model_path, yolo::Type::V8, confidence_threshold, nms_threshold);
	if (yolo == nullptr) return;

	auto objs = yolo->forward(cvimg(image));

	int i = 0;
	for (auto &obj : objs) {
		uint8_t b, g, r;
		tie(b, g, r) = yolo::random_color(obj.class_label);
		cv::rectangle(image, cv::Point(obj.left, obj.top), cv::Point(obj.right, obj.bottom),
			cv::Scalar(b, g, r), 5);

		auto name = cocolabels[obj.class_label];
		auto caption = cv::format("%s %.2f", name, obj.confidence);
		int width = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
		cv::rectangle(image, cv::Point(obj.left - 3, obj.top - 33),
			cv::Point(obj.left + width, obj.top), cv::Scalar(b, g, r), -1);
		cv::putText(image, caption, cv::Point(obj.left, obj.top - 5), 0, 1, cv::Scalar::all(0), 2, 16);

		if (obj.seg) {
			cv::imwrite(cv::format("%d_mask.jpg", i),
				cv::Mat(obj.seg->height, obj.seg->width, CV_8U, obj.seg->data));
			i++;
		}
	}

	printf("Save result to Result.jpg, %d objects\n", (int)objs.size());
	cv::imwrite("Result.jpg", image);
}

int main() {
	std::string img_path = "zidane.jpg";
	std::string model_path = "yolov8n_fp16.trt";
	single_inference(img_path, model_path);
	return 0;
}

extern "C" __declspec(dllexport) void* Init(const char* model_path) {
	std::string engine_name = model_path;
	float confidence_threshold = 0.25f;
	float nms_threshold = 0.5f;
	auto yolo = loadraw(engine_name, yolo::Type::V8, confidence_threshold, nms_threshold);
	return yolo;
}

extern "C" __declspec(dllexport) void Detect(void* p, int rows, int cols, unsigned char* src_data, float(*res_array)[6]) {
	cv::Mat frame = cv::Mat(rows, cols, CV_8UC3, src_data);
	yolo::Infer* yolov8 = (yolo::Infer*)p;
	auto objs = yolov8->forward(cvimg(frame));
	int i = 0;
	for (auto &obj : objs) {
		res_array[i][0] = obj.left;
		res_array[i][1] = obj.top;
		res_array[i][2] = obj.right - obj.left;
		res_array[i][3] = obj.bottom - obj.top;;
		res_array[i][4] = obj.class_label;
		res_array[i][5] = obj.confidence;
		i++;
	}
}