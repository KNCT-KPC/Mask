#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

const String files[] = {
	/* �����ɉ摜�t�@�C����ǉ� */
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

	// �摜��ǂݍ���
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

	// �E�B���h�E����
	namedWindow(winName, WINDOW_AUTOSIZE);

	// �J�X�P�[�h���ފ�̎擾
	CascadeClassifier cascade;
	if (!cascade.load("haarcascade_frontalface_alt.xml")) return -1;

	// ���`�̎擾
	vector<Rect> faces;

	// �J�����̃L���v�`��
	VideoCapture cap(0);
	if (!cap.isOpened()) return -1;

	while (1) {

		// �J�����f���̎擾	
		cap >> frame;

		// �O���[�X�P�[���ϊ�
		cvtColor(frame, grayframe, CV_RGB2GRAY);

		// �q�X�g�O�����̕��R��
		equalizeHist(grayframe, grayframe);

		// �J�X�P�[�h���ފ�Ŋ�̒T��
		cascade.detectMultiScale(grayframe, faces, 1.2, 2, CV_HAAR_SCALE_IMAGE, Size(30, 30));

		for (vector<Rect>::const_iterator r = faces.begin(); r != faces.end(); ++r) {
			double rw = r->width;
			double rh = maskImg.rows*r->width / maskImg.cols;
			double rx = r->x;
			double ry = r->y;

			// �摜�����T�C�Y����
			Mat remaskImg(rh, rw, maskImg.channels());
			resize(maskImg, remaskImg, remaskImg.size(), 0, 0, INTER_LINEAR);

			for (int x = rx; x < rx + rw; x++) {
				for (int y = ry; y < ry + rh; y++) {
					if (x < 0 || y < 0 || x >= frame.cols || y >= frame.rows) continue;
					// ��f
					int z1 = (y - ry)*remaskImg.step + (x - rx)*remaskImg.channels();
					int r = remaskImg.data[z1 + 2];
					int g = remaskImg.data[z1 + 1];
					int b = remaskImg.data[z1];
					int a = remaskImg.data[z1 + 3];
					// ���ߓx���l��������������
					if (a) {
						int z2 = y *frame.step + x * frame.channels();
						frame.data[z2 + 2] = r;
						frame.data[z2 + 1] = g;
						frame.data[z2] = b;
					}
				}
			}
		}
		// �f���̕\��
		imshow(winName, frame);

		int key = waitKey(1);
		if (key == 113) { // q�������ꂽ�Ƃ�
			break; // while���[�v���甲����D
		}
		else if (key == 99) { // c�������ꂽ�Ƃ�
			destroyAllWindows();
			// �摜��ǂݍ���
			if ((maskImg = imread(files[++filenum %= array_length(files)], IMREAD_UNCHANGED)).empty()) return -1;
			namedWindow(winName, WINDOW_AUTOSIZE);
		}
		else if (key == 115) { // s�������ꂽ�Ƃ�
			imwrite("img.png", frame); // �t���[���摜��ۑ�����D
		}
	}
	return 1;
}