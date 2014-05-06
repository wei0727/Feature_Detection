#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\features2d\features2d.hpp>
#include <opencv2\nonfree\nonfree.hpp>
#include <opencv2\calib3d\calib3d.hpp>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
using namespace std ;
using namespace cv ;

bool dmatch_cmp(DMatch d1, DMatch d2){
	return d1.distance < d2.distance ;
}

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

void keypoints2points(vector<KeyPoint> &kps, vector<Point2d> &pts){
	for(int i=0; i<kps.size(); i++)
		pts.push_back(kps[i].pt) ;
}

int inliers(vector<Point2d> &pts_obj, vector<Point2d> &pts_target, vector<vector<DMatch>> &matches, Mat &transform_mat, double threshold){
	vector<Point2d> transform_obj ;
	transform_obj.reserve(pts_obj.size()) ;
	perspectiveTransform(pts_obj, transform_obj, transform_mat) ;
	int inliers = 0 ;
	for(int i=0; i<transform_obj.size(); i++){
		for(int k=0; k<matches[i].size(); k++){
			Point2d dif = transform_obj[i] - pts_target[matches[i][k].trainIdx] ;
			double d = dif.x*dif.x + dif.y*dif.y ;
			if(d < threshold){
				inliers += 1 ;
				break ;
			}
		}
	}
	return inliers ;
}

int main(){
	IplImage *img = cvLoadImage("object/a2.jpg") ;
	IplImage *target = cvLoadImage("target/target1.jpg") ;
	//Mat img = imread("a1.jpg", 0) ;
	Mat kimg_obj(img) ;
	Mat kimg_target(target) ;
	//initModule_nonfree() ;
	SiftFeatureDetector detector(0.05, 5.0) ;
	vector<KeyPoint> kps_obj, kps_target ;
	vector<Point2d> pts_obj, pts_target ;
	detector.detect(kimg_obj, kps_obj) ;
	detector.detect(kimg_target, kps_target) ;
	//cout << kps_obj.size() << " " << kps_target.size() << endl ;
	keypoints2points(kps_obj, pts_obj) ;
	keypoints2points(kps_target, pts_target) ;
	
	drawKeypoints(img, kps_obj, kimg_obj, Scalar(255, 255, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS) ;
	drawKeypoints(target, kps_target, kimg_target, Scalar(255, 255, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS) ;
	
	SiftDescriptorExtractor extractor ;
	Mat descriptor_obj, descriptor_target ;
	extractor.compute(img, kps_obj, descriptor_obj) ;
	extractor.compute(target, kps_target, descriptor_target) ;

	//knn match
	const int num_knn = 4 ;
	BFMatcher matcher(NORM_L2) ;
	vector<vector<DMatch>> matches(pts_obj.size()) ;
	//matcher.knnMatch(descriptor_obj, descriptor_target, matches, num_knn) ;
	Mat distance ;
	for(int i=0; i<kps_obj.size(); i++){
		for(int k=0; k<kps_target.size(); k++){
			double distance = norm(descriptor_obj.row(0), descriptor_target.row(k)) ;
			if(matches[i].size()==0)
				matches[i].push_back(DMatch(i, k, distance)) ;
			else{
				matches[i].insert(lower_bound(matches[i].begin(), matches[i].end(), DMatch(i, k, distance), dmatch_cmp), DMatch(i, k, distance)) ;
				if(matches[i].size()>4){
					matches[i].pop_back() ;
				}
			}
		}
	}
	//RANSAC
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
			setPtsMat(kps_obj[rndpts[0]], kps_target[matches[rndpts[0]][i].trainIdx], pts_mat.ptr<double>(0), pts_mat.ptr<double>(1)) ;
			for(int j=0; j<num_knn; j++){
				setPtsMat(kps_obj[rndpts[1]], kps_target[matches[rndpts[1]][j].trainIdx], pts_mat.ptr<double>(2), pts_mat.ptr<double>(3)) ;
				for(int k=0; k<num_knn; k++){
					setPtsMat(kps_obj[rndpts[2]], kps_target[matches[rndpts[2]][k].trainIdx], pts_mat.ptr<double>(4), pts_mat.ptr<double>(5)) ;
					for(int m=0; m<num_knn; m++){
						setPtsMat(kps_obj[rndpts[3]], kps_target[matches[rndpts[3]][m].trainIdx], pts_mat.ptr<double>(6), pts_mat.ptr<double>(7)) ;
						//cout << pts_mat << endl ;
						Mat utu = pts_mat.t() * pts_mat ;
						Mat eval(1, 9, CV_64F) ;
						Mat evec(9, 9, CV_64F) ;
						eigen(utu, eval, evec) ;
						//cout << norm(pts_mat*evec.row(0).t()) << endl ;
						//cout << pts_mat*evec.row(8).t() << endl ;
						//cout << norm(pts_mat*evec.row(8).t()) << endl ;
						//cout << evec.row(8) << endl ;
						Mat transform_mat(3, 3, CV_64F, evec.ptr<double>(8)) ;
						//cout << transform_mat << endl ;
						//////////////////////////////////
						vector<Point2d> tmp, tmp2 ;
						tmp.push_back(pts_obj[rndpts[0]]) ;
						tmp.push_back(pts_obj[rndpts[1]]) ;
						tmp.push_back(pts_obj[rndpts[2]]) ;
						tmp.push_back(pts_obj[rndpts[3]]) ;
						perspectiveTransform(tmp, tmp2, transform_mat) ;
						Point2d dif0 = tmp2[0] - pts_target[matches[rndpts[0]][i].trainIdx] ;
						Point2d dif1 = tmp2[1] - pts_target[matches[rndpts[1]][j].trainIdx] ;
						Point2d dif2 = tmp2[2] - pts_target[matches[rndpts[2]][k].trainIdx] ;
						Point2d dif3 = tmp2[3] - pts_target[matches[rndpts[3]][m].trainIdx] ;
						//cout << pts_target[matches[rndpts[0]][i].trainIdx] << endl ;
						//cout << pts_target[matches[rndpts[1]][j].trainIdx] << endl ;
						//cout << pts_target[matches[rndpts[2]][k].trainIdx] << endl ;
						//cout << pts_target[matches[rndpts[3]][m].trainIdx] << endl ;
						//cout << tmp[0] * pts_target[matches[rndpts[0]][i].trainIdx].x << endl ;
						//cout << tmp[0] * pts_target[matches[rndpts[0]][i].trainIdx].y << endl ;
						//cout << tmp[1] * pts_target[matches[rndpts[1]][j].trainIdx].x << endl ;
						//cout << tmp[1] * pts_target[matches[rndpts[1]][j].trainIdx].y << endl ;
						//cout << tmp[2] * pts_target[matches[rndpts[2]][k].trainIdx].x << endl ;
						//cout << tmp[2] * pts_target[matches[rndpts[2]][k].trainIdx].y << endl ;
						//cout << tmp[3] * pts_target[matches[rndpts[3]][m].trainIdx].x << endl ;
						//cout << tmp[3] * pts_target[matches[rndpts[3]][m].trainIdx].y << endl ;
						//for(int a=0; a<4; a++){
						//	cout << tmp[a] << "\t" << tmp2[a] << endl ;
						//}
						//Mat p1 = (Mat_<double>(3, 1) << tmp[0].x, tmp[0].y, 1) ;
						//cout << p1 << endl ;
						//cout << transform_mat*p1 << endl ;
						double d0 = dif0.x*dif0.x + dif0.y*dif0.y ;
						double d1 = dif1.x*dif1.x + dif1.y*dif1.y ;
						double d2 = dif2.x*dif2.x + dif2.y*dif2.y ;
						double d3 = dif3.x*dif3.x + dif3.y*dif3.y ;
						double sum = d0+d1+d2+d3 ;
						if(sum < 60000){
 							cout << transform_mat << endl ;
							cout << sum << endl ;
							cout << pts_target[matches[rndpts[0]][i].trainIdx] << endl ;
							cout << pts_target[matches[rndpts[1]][j].trainIdx] << endl ;
							cout << pts_target[matches[rndpts[2]][k].trainIdx] << endl ;
							cout << pts_target[matches[rndpts[3]][m].trainIdx] << endl ;
							for(int a=0; a<4; a++){
								cout << tmp[a] << "\t" << tmp2[a] << endl ;
							}
							vector<DMatch> tmp_matches ;
							tmp_matches.push_back(matches[rndpts[0]][i]) ;
							tmp_matches.push_back(matches[rndpts[1]][j]) ;
							tmp_matches.push_back(matches[rndpts[2]][k]) ;
							tmp_matches.push_back(matches[rndpts[3]][m]) ;
							for(int a=0; a<4; a++)
								cout << tmp_matches[a].queryIdx << "\t" << tmp_matches[a].trainIdx << endl ;
							Mat img_matches ;
							drawMatches(Mat(img), kps_obj, Mat(target), kps_target, tmp_matches, img_matches) ;
							imshow("matches", img_matches) ;
							cvWaitKey(0) ;
						}
						//////////////////////////////////
						//int num_inliers = inliers(pts_obj, pts_target, matches, transform_mat, 5.0) ;
						//if(num_inliers>0)
							//cout << pts_obj.size() << "\t" << num_inliers << endl ;
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