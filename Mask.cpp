#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

const String files[] = {
	/* ここに画像ファイルを追加 */
	"laugh.png",
	"guyfawkes.png",
	"ago.png"
};
const char winName[] = "Mask";

template < typename TYPE, std::size_t SIZE >
std::size_t array_length(const TYPE(&)[SIZE])
{
	return SIZE;
}

int main(int argc, char *argv[])
{
	Mat maskImg, frame, grayframe;
	int filenum = 0;

	// 画像を読み込む
	if ((maskImg = imread(files[filenum], IMREAD_UNCHANGED)).empty())
	{
		printf("-----------------------------\n");
		printf("       image not exist\n");
		printf("-----------------------------\n");
		return -1;
	}
	else
	{
		printf("-----------------------------\n");
		printf("    Press ANY key to quit\n");
		printf("-----------------------------\n");
	}

	// ウィンドウ生成
	namedWindow(winName, WINDOW_AUTOSIZE);

	// カスケード分類器の取得
	CascadeClassifier cascade;
	if (!cascade.load("haarcascade_frontalface_alt.xml")) return -1;

	// 顔矩形の取得
	vector<Rect> faces;

	// カメラのキャプチャ
	VideoCapture cap(0);
	if (!cap.isOpened()) return -1;

	while (1) {

		// カメラ映像の取得	
		cap >> frame;

		// グレースケール変換
		cvtColor(frame, grayframe, CV_RGB2GRAY);

		// ヒストグラムの平坦化
		equalizeHist(grayframe, grayframe);

		// カスケード分類器で顔の探索
		cascade.detectMultiScale(grayframe, faces, 1.2, 2, CV_HAAR_SCALE_IMAGE, Size(30, 30));

		for (vector<Rect>::const_iterator r = faces.begin(); r != faces.end(); ++r) {
			double rw = r->width;
			double rh = maskImg.rows*r->width / maskImg.cols;
			double rx = r->x;
			double ry = r->y;

			// 画像をリサイズする
			Mat remaskImg(rh, rw, maskImg.channels());
			resize(maskImg, remaskImg, remaskImg.size(), 0, 0, INTER_LINEAR);

			for (int x = rx; x < rx + rw; x++) {
				for (int y = ry; y < ry + rh; y++) {
					if (x < 0 || y < 0 || x >= frame.cols || y >= frame.rows) continue;
					// 画素
					int z1 = (y - ry)*remaskImg.step + (x - rx)*remaskImg.channels();
					int r = remaskImg.data[z1 + 2];
					int g = remaskImg.data[z1 + 1];
					int b = remaskImg.data[z1];
					int a = remaskImg.data[z1 + 3];
					// 透過度を考慮した書き換え
					if (a) {
						int z2 = y *frame.step + x * frame.channels();
						frame.data[z2 + 2] = r;
						frame.data[z2 + 1] = g;
						frame.data[z2] = b;
					}
				}
			}
		}
		// 映像の表示
		imshow(winName, frame);

		int key = waitKey(1);
		if (key == 113) { // qが押されたとき
			break; // whileループから抜ける．
		}
		else if (key == 99) { // cが押されたとき
			destroyAllWindows();
			// 画像を読み込む
			if ((maskImg = imread(files[++filenum %= array_length(files)], IMREAD_UNCHANGED)).empty()) return -1;
			namedWindow(winName, WINDOW_AUTOSIZE);
		}
		else if (key == 115) { // sが押されたとき
			imwrite("img.png", frame); // フレーム画像を保存する．
		}
	}
	return 1;
}