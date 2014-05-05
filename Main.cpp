#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\features2d\features2d.hpp>
#include <opencv2\nonfree\nonfree.hpp>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
using namespace std ;
using namespace cv ;

void setPtsMat(KeyPoint obj, KeyPoint target, double *p1, double *p2){
	p1[0] = obj.pt.x ;
	p1[1] = obj.pt.y ;
	p1[2] = 1 ;
	p1[6] = -p1[0]*target.pt.x ;
	p1[7] = -p1[1]*target.pt.x ;
	p1[8] = -target.pt.x ;
	
	p2[3] = obj.pt.x ;
	p2[4] = obj.pt.y ;
	p2[5] = 1 ;
	p2[6] = -p2[0]*target.pt.y ;
	p2[7] = -p2[1]*target.pt.y ;
	p2[8] = -target.pt.y ;
}

int main(){
	IplImage *img = cvLoadImage("object/a1.jpg") ;
	IplImage *target = cvLoadImage("target/target1.jpg") ;
	//Mat img = imread("a1.jpg", 0) ;
	Mat kimg_obj(img) ;
	Mat kimg_target(target) ;
	//initModule_nonfree() ;
	SiftFeatureDetector detector(0.05, 5.0) ;
	vector<KeyPoint> kps_obj, kps_target ;
	detector.detect(kimg_obj, kps_obj) ;
	detector.detect(kimg_target, kps_target) ;
	
	drawKeypoints(img, kps_obj, kimg_obj, Scalar(255, 255, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS) ;
	drawKeypoints(target, kps_target, kimg_target, Scalar(255, 255, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS) ;
	
	SiftDescriptorExtractor extractor ;
	Mat descriptor_obj, descriptor_target ;
	extractor.compute(img, kps_obj, descriptor_obj) ;
	extractor.compute(target, kps_target, descriptor_target) ;


	const int num_knn = 4 ;
	BFMatcher matcher(NORM_L2) ;
	vector<vector<DMatch>> matches ;
	matcher.knnMatch(descriptor_obj, descriptor_target, matches, num_knn) ;
	
	srand(time(NULL)) ;
	while(true){
		int rndpts[4] ;
		for(int i=0; i<4; i++){
			while(true){
				bool repete=false ;
				rndpts[i] = rand()%kps_obj.size() ;
				for(int j=0; j<i; j++){
					if(rndpts[i] == rndpts[j]){
						repete = true ;
						break ;
					}
				}
				if(!repete)
					break ;
			}
		}
		Mat pts_mat = Mat::zeros(8, 9, CV_64F) ;
		double *p ;
		for(int i=0; i<num_knn; i++){
			setPtsMat(kps_obj[rndpts[0]], kps_target[i], pts_mat.ptr<double>(0), pts_mat.ptr<double>(1)) ;
			for(int j=0; j<num_knn; j++){
				setPtsMat(kps_obj[rndpts[1]], kps_target[j], pts_mat.ptr<double>(2), pts_mat.ptr<double>(3)) ;
				for(int k=0; k<num_knn; k++){
					setPtsMat(kps_obj[rndpts[2]], kps_target[k], pts_mat.ptr<double>(4), pts_mat.ptr<double>(5)) ;
					for(int m=0; m<num_knn; m++){
						setPtsMat(kps_obj[rndpts[3]], kps_target[m], pts_mat.ptr<double>(6), pts_mat.ptr<double>(7)) ;
						Mat utu = pts_mat.t() * pts_mat ;
						Mat eval(1, 9, CV_64F) ;
						Mat evec(9, 9, CV_64F) ;
						eigen(utu, eval, evec) ;
						cout << norm(pts_mat*evec.row(0).t()) << endl ;
						cout << norm(pts_mat*evec.row(8).t()) << endl ;
					}
				}
			}
		}
	}
	cvShowImage("kimg_target", &IplImage(kimg_target)) ;
	cvShowImage("kimg_object", &IplImage(kimg_obj)) ;
	cvWaitKey(0) ;
	return 0 ;
}