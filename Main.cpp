#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\nonfree\features2d.hpp>
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
	p2[6] = -p2[3]*target.pt.y ;
	p2[7] = -p2[4]*target.pt.y ;
	p2[8] = -target.pt.y ;
}

void setPtsMat(Point2d obj, Point2d target, double *p1, double *p2){
	p1[0] = obj.x ;
	p1[1] = obj.y ;
	p1[2] = 1 ;
	p1[6] = -p1[0]*target.x ;
	p1[7] = -p1[1]*target.x ;
	p1[8] = -target.x ;
	
	p2[3] = obj.x ;
	p2[4] = obj.y ;
	p2[5] = 1 ;
	p2[6] = -p2[0]*target.y ;
	p2[7] = -p2[1]*target.y ;
	p2[8] = -target.y ;
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
	for(int i=0; i<matches.size(); i++){
		for(int k=0; k<matches[i].size(); k++){
			Point2d dif = transform_obj[matches[i][k].queryIdx] - pts_target[matches[i][k].trainIdx] ;
			double d = sqrt(dif.x*dif.x + dif.y*dif.y) ;
			if(d < threshold){
				inliers += 1 ;
				break ;
			}
		}
	}
	return inliers ;
}

void normalized_dlt(vector<Point2d> &obj, vector<Point2d> &target, Mat &transform_mat){
	Mat t_obj = Mat::eye(3, 3, CV_64F) ;
	Mat t_target = Mat::eye(3, 3, CV_64F) ;
	Point2d mean_obj(0, 0), mean_target(0, 0) ;
	for(int i=0; i<obj.size(); i++){
		//cout << obj[i] << "\t" << target[i] << endl ;
		mean_obj += obj[i] ;
		mean_target += target[i] ;
	}
	mean_obj.x /= obj.size() ;
	mean_obj.y /= obj.size() ;
	mean_target.x /= target.size() ;
	mean_target.y /= target.size() ;
	double s_obj=0, s_target=0 ;
	for(int i=0; i<obj.size(); i++){
		s_obj += norm(obj[i] - mean_obj) ;
		s_target += norm(target[i] - mean_target) ;
	}
	s_obj = sqrt(2.0)*obj.size()/s_obj ;
	s_target = sqrt(2.0)*obj.size()/s_target ;
	t_obj.at<double>(0, 0) = s_obj ;
	t_obj.at<double>(0, 2) = -s_obj * mean_obj.x ;
	t_obj.at<double>(1, 1) = s_obj ;
	t_obj.at<double>(1, 2) = -s_obj * mean_obj.y ;
	t_target.at<double>(0, 0) = s_target ;
	t_target.at<double>(0, 2) = -s_target * mean_target.x ;
	t_target.at<double>(1, 1) = s_target ;
	t_target.at<double>(1, 2) = -s_target * mean_target.y ;
	vector<Point2d> pts_t_obj, pts_t_target ;
	//cout << t_obj << endl ;
	//cout << t_target << endl ;
	perspectiveTransform(obj, pts_t_obj, t_obj) ;
	perspectiveTransform(target, pts_t_target, t_target) ;
	//cout << pts_t_obj.size() << endl ;
	Mat u = Mat::zeros(8, 9, CV_64F) ;
	for(int i=0; i<pts_t_obj.size(); i++){
		//cout << i << "\t" ;
		//cout << pts_t_obj[i] << "\t" << pts_t_target[i] << endl ;
		setPtsMat(pts_t_obj[i], pts_t_target[i], u.ptr<double>(i*2), u.ptr<double>(i*2+1)) ;
	}
	Mat utu = u.t() * u ;
	Mat eval(1, 9, CV_64F) ;
	Mat evec(9, 9, CV_64F) ;
	eigen(utu, eval, evec) ;
	Mat t_mat(3, 3, CV_64F, evec.ptr<double>(8)) ;
	transform_mat = t_obj.inv() * t_mat * t_target ;

	//cout << utu << endl ;
}

int main(){
	IplImage *img = cvLoadImage("object/b1.jpg") ;
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
	keypoints2points(kps_obj, pts_obj) ;
	keypoints2points(kps_target, pts_target) ;
	//drawKeypoints(img, kps_obj, kimg_obj, Scalar(255, 255, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS) ;
	//drawKeypoints(target, kps_target, kimg_target, Scalar(255, 255, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS) ;
	
	SiftDescriptorExtractor extractor ;
	Mat descriptor_obj, descriptor_target ;
	extractor.compute(img, kps_obj, descriptor_obj) ;
	extractor.compute(target, kps_target, descriptor_target) ;

	//knn match
	const int num_knn = 4 ;
	BFMatcher matcher(NORM_L2) ;
	vector<vector<DMatch>> matches(pts_obj.size()), good_matches ;
	//matcher.knnMatch(descriptor_obj, descriptor_target, matches, num_knn) ;
	Mat distance ;
	double mind=1000 ;
	for(int i=0; i<kps_obj.size(); i++){
		for(int k=0; k<kps_target.size(); k++){
			double distance = norm(descriptor_obj.row(i), descriptor_target.row(k)) ;
			mind = min(mind, distance) ;
			if(matches[i].size()==0)
				matches[i].push_back(DMatch(i, k, distance)) ;
			else{
				matches[i].insert(lower_bound(matches[i].begin(), matches[i].end(), DMatch(i, k, distance), dmatch_cmp), DMatch(i, k, distance)) ;
				if(matches[i].size()>num_knn){
					matches[i].pop_back() ;
				}
			}
		}
	}
	cout << mind << endl ;
	mind *= 3 ;
	if(mind == 0)
		mind = 100 ;
	for(int i=0; i<kps_obj.size(); i++){
		vector<DMatch> tmp ;
		for(int k=0; k<kps_target.size(); k++){
			if(matches[i][k].distance <= mind)
				tmp.push_back(matches[i][k]) ;
			else
				break ;
		}
		if(!tmp.empty())
			good_matches.push_back(tmp) ;
	}
	cout << good_matches.size() << endl ;
	//good_matches = matches ;
	//RANSAC
	srand(time(NULL)) ;
	while(true){
		int rndpts[4] ;
		for(int i=0; i<4; i++){
			while(true){
				bool repete=false ;
				rndpts[i] = rand()%good_matches.size() ;
				for(int j=0; j<i; j++){
					if(pts_obj[good_matches[rndpts[i]][0].queryIdx] == pts_obj[good_matches[rndpts[j]][0].queryIdx]){
						repete = true ;
						break ;
					}
				}
				if(!repete)
					break ;
			}
		}
		
		Mat pts_mat = Mat::zeros(8, 9, CV_64F) ;
		vector<Point2d> tmp_obj(4), tmp_target(4) ;
		for(int i=0; i<good_matches[rndpts[0]].size(); i++){
			setPtsMat(kps_obj[good_matches[rndpts[0]][i].queryIdx], kps_target[good_matches[rndpts[0]][i].trainIdx], pts_mat.ptr<double>(0), pts_mat.ptr<double>(1)) ;
			tmp_obj[0] = pts_obj[good_matches[rndpts[0]][i].queryIdx] ;
			tmp_target[0] = pts_target[good_matches[rndpts[0]][i].trainIdx] ;
			for(int j=0; j<good_matches[rndpts[1]].size(); j++){
				if(pts_target[good_matches[rndpts[1]][j].trainIdx] == pts_target[good_matches[rndpts[0]][i].trainIdx])
					continue ;
				setPtsMat(kps_obj[good_matches[rndpts[1]][j].queryIdx], kps_target[good_matches[rndpts[1]][j].trainIdx], pts_mat.ptr<double>(2), pts_mat.ptr<double>(3)) ;
				tmp_obj[1] = pts_obj[good_matches[rndpts[1]][j].queryIdx] ;
				tmp_target[1] = pts_target[good_matches[rndpts[1]][j].trainIdx] ;
				for(int k=0; k<good_matches[rndpts[2]].size(); k++){
					if(pts_target[good_matches[rndpts[2]][k].trainIdx] == pts_target[good_matches[rndpts[0]][i].trainIdx] ||
						pts_target[good_matches[rndpts[2]][k].trainIdx] == pts_target[good_matches[rndpts[1]][j].trainIdx])
						continue ;
					setPtsMat(kps_obj[good_matches[rndpts[2]][k].queryIdx], kps_target[good_matches[rndpts[2]][k].trainIdx], pts_mat.ptr<double>(4), pts_mat.ptr<double>(5)) ;
					tmp_obj[2] = pts_obj[good_matches[rndpts[2]][k].queryIdx] ;
					tmp_target[2] = pts_target[good_matches[rndpts[2]][k].trainIdx] ;
					for(int m=0; m<good_matches[rndpts[3]].size(); m++){
						if(pts_target[good_matches[rndpts[3]][m].trainIdx] == pts_target[good_matches[rndpts[0]][i].trainIdx] ||
							pts_target[good_matches[rndpts[3]][m].trainIdx] == pts_target[good_matches[rndpts[1]][j].trainIdx] ||
							pts_target[good_matches[rndpts[3]][m].trainIdx] == pts_target[good_matches[rndpts[2]][k].trainIdx])
							continue ;
						setPtsMat(kps_obj[good_matches[rndpts[3]][m].queryIdx], kps_target[good_matches[rndpts[3]][m].trainIdx], pts_mat.ptr<double>(6), pts_mat.ptr<double>(7)) ;
						tmp_obj[3] = pts_obj[good_matches[rndpts[3]][m].queryIdx] ;
						tmp_target[3] = pts_target[good_matches[rndpts[3]][m].trainIdx] ;
						Mat utu = pts_mat.t() * pts_mat ;
						Mat eval(1, 9, CV_64F) ;
						Mat evec(9, 9, CV_64F) ;
						eigen(utu, eval, evec) ;
						Mat transform_mat(3, 3, CV_64F, evec.ptr<double>(8)) ;
						//Mat transform_mat ;
						//normalized_dlt(tmp_obj, tmp_target, transform_mat) ;
						//////////////////////////////////
						vector<Point2d> tmp, tmp2 ;
						tmp.push_back(pts_obj[good_matches[rndpts[0]][i].queryIdx]) ;
						tmp.push_back(pts_obj[good_matches[rndpts[1]][j].queryIdx]) ;
						tmp.push_back(pts_obj[good_matches[rndpts[2]][k].queryIdx]) ;
						tmp.push_back(pts_obj[good_matches[rndpts[3]][m].queryIdx]) ;
						perspectiveTransform(tmp, tmp2, transform_mat) ;
						Point2d dif0 = tmp2[0] - pts_target[good_matches[rndpts[0]][i].trainIdx] ;
						Point2d dif1 = tmp2[1] - pts_target[good_matches[rndpts[1]][j].trainIdx] ;
						Point2d dif2 = tmp2[2] - pts_target[good_matches[rndpts[2]][k].trainIdx] ;
						Point2d dif3 = tmp2[3] - pts_target[good_matches[rndpts[3]][m].trainIdx] ;
						double d0 = sqrt(dif0.x*dif0.x + dif0.y*dif0.y) ;
						double d1 = sqrt(dif1.x*dif1.x + dif1.y*dif1.y) ;
						double d2 = sqrt(dif2.x*dif2.x + dif2.y*dif2.y) ;
						double d3 = sqrt(dif3.x*dif3.x + dif3.y*dif3.y) ;
						double sum = d0+d1+d2+d3 ;
						//cout << sum << endl ;
						if(sum < 100){
							//cout << utu << endl ;
							cout << eval << endl ;
 							cout << transform_mat << endl ;
							cout << sum << endl ;
							cout << pts_target[good_matches[rndpts[0]][i].trainIdx] << endl ;
							cout << pts_target[good_matches[rndpts[1]][j].trainIdx] << endl ;
							cout << pts_target[good_matches[rndpts[2]][k].trainIdx] << endl ;
							cout << pts_target[good_matches[rndpts[3]][m].trainIdx] << endl ;
							for(int a=0; a<4; a++){
								cout << tmp[a] << "\t" << tmp2[a] << endl ;
							}
							vector<KeyPoint> tmp_kps1, tmp_kps2 ;
							vector<DMatch> tmp_matches ;
							tmp_kps1.push_back(kps_obj[good_matches[rndpts[0]][i].queryIdx]) ;
							tmp_kps1.push_back(kps_obj[good_matches[rndpts[1]][j].queryIdx]) ;
							tmp_kps1.push_back(kps_obj[good_matches[rndpts[2]][k].queryIdx]) ;
							tmp_kps1.push_back(kps_obj[good_matches[rndpts[3]][m].queryIdx]) ;
							tmp_kps2.push_back(kps_target[good_matches[rndpts[0]][i].trainIdx]) ;
							tmp_kps2.push_back(kps_target[good_matches[rndpts[1]][j].trainIdx]) ;
							tmp_kps2.push_back(kps_target[good_matches[rndpts[2]][k].trainIdx]) ;
							tmp_kps2.push_back(kps_target[good_matches[rndpts[3]][m].trainIdx]) ;
							tmp_matches.push_back(DMatch(0, 0, 0)) ;
							tmp_matches.push_back(DMatch(1, 1, 0)) ;
							tmp_matches.push_back(DMatch(2, 2, 0)) ;
							tmp_matches.push_back(DMatch(3, 3, 0)) ;
							Mat img_matches ;
							drawMatches(Mat(img), tmp_kps1, Mat(target), tmp_kps2, tmp_matches, img_matches) ;
							for(int a=0; a<4; a++)
								circle(img_matches, cvPoint((img->width)+tmp2[a].x, tmp2[a].y), 5, Scalar(255, 0, 0), 2) ;
							imshow("matches", img_matches) ;
							//cvWaitKey(0) ;
							int num_inliers = inliers(pts_obj, pts_target, good_matches, transform_mat, 60.0) ;
							cout << good_matches.size() << "\t" << num_inliers << endl ;
							if(num_inliers>40){
								vector<Point2d> obj_corner(4, Point2d(0, 0)), target_corner ;
								obj_corner[1] = Point2d(img->width, 0) ;
								obj_corner[2] = Point2d(img->width, img->height) ;
								obj_corner[3] = Point2d(0, img->height) ;
								perspectiveTransform(obj_corner, target_corner, transform_mat) ;
								Mat tmp = Mat(target).clone() ;
								for(int a=0; a<4; a++)
									circle(tmp, target_corner[a], 5, Scalar(255, 255, 0), 2) ;
								line(tmp, target_corner[0], target_corner[1], Scalar(0, 255, 0)) ;
								line(tmp, target_corner[1], target_corner[2], Scalar(0, 255, 0)) ;
								line(tmp, target_corner[2], target_corner[3], Scalar(0, 255, 0)) ;
								line(tmp, target_corner[3], target_corner[0], Scalar(0, 255, 0)) ;
								imshow("tmp", tmp) ;
								cvWaitKey(0) ;
								cvDestroyAllWindows() ;
							}
						}
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