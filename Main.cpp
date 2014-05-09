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
	p2[6] = -p2[3]*target.y ;
	p2[7] = -p2[4]*target.y ;
	p2[8] = -target.y ;
}

void keypoints2points(vector<KeyPoint> &kps, vector<Point2d> &pts){
	for(int i=0; i<kps.size(); i++)
		pts.push_back(kps[i].pt) ;
}

int inliers(vector<Point2d> &pts_obj, vector<Point2d> &pts_target, vector<vector<DMatch>> &matches, Mat &transform_mat, double threshold, double &total_error){
	vector<Point2d> transform_obj ;
	transform_obj.reserve(pts_obj.size()) ;
	perspectiveTransform(pts_obj, transform_obj, transform_mat) ;
	int inliers = 0 ;
	int total_num = 0 ;
	total_error = 0 ;
	for(int i=0; i<matches.size(); i++){
		for(int k=0; k<matches[i].size(); k++){
			Point2d dif = transform_obj[matches[i][k].queryIdx] - pts_target[matches[i][k].trainIdx] ;
			double d = sqrt(dif.x*dif.x + dif.y*dif.y) ;
			total_error += d ;
			total_num ++ ;
			if(d < threshold){
				inliers += 1 ;
				break ;
			}
		}
	}
	total_error /= total_num ;
	return inliers ;
}

bool niceHomography(Mat &h){
	double *r0 = h.ptr<double>(0) ;
	double *r1 = h.ptr<double>(1) ;
	double *r2 = h.ptr<double>(2) ;

	double det = r0[0]*r1[1] - r1[0]*r0[1] ;
	if(det < 0)
		return false ;

	double n1 = sqrt(r0[0]*r0[0] + r1[0]*r1[0]) ;
	cout << "n1 " << n1 << endl ;
	//if(n1>4 || n1<0.1)
	//	return false ;

	double n2 = sqrt(r0[1]*r0[1] + r1[1]*r1[1]) ;
	cout << "n2 " << n2 << endl ;
	//if(n2>4 || n2<0.1)
	//	return false ;

	double n3 = sqrt(r2[0]*r2[0] + r2[1]*r2[1]) ;
	cout << "n3 " << n3 << endl ;
	//if(n3>4 || n3<0.1)
	//	return false ;
	return true ;
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

void knn_match(int k, Mat &descriptor_obj, Mat &descriptor_target, vector<vector<DMatch>> &matches, vector<vector<DMatch>> &good_matches){
	double mind = DBL_MAX ;
	double maxd = 0 ;
	double mean = 0 ;
	for(int i=0; i<descriptor_obj.rows; i++){
		for(int k=0; k<descriptor_target.rows; k++){
			double distance = norm(descriptor_obj.row(i), descriptor_target.row(k)) ;
			mind = min(mind, distance) ;
			maxd = max(maxd, distance) ;
			mean += distance ;
			if(matches[i].size()==0)
				matches[i].push_back(DMatch(i, k, distance)) ;
			else{
				matches[i].insert(lower_bound(matches[i].begin(), matches[i].end(), DMatch(i, k, distance), dmatch_cmp), DMatch(i, k, distance)) ;
				if(matches[i].size()>k){
					matches[i].pop_back() ;
				}
			}
		}
	}
	mean = mean/(descriptor_obj.rows*descriptor_target.rows) ;
	cout << "min " << mind << endl ;
	cout << "max " << maxd << endl ;
	cout << "mean " << mean << endl ;
	mind *= 3 ;
	if(mind == 0){
		mind = maxd/3 ;
	}
	for(int i=0; i<descriptor_obj.rows; i++){
		vector<DMatch> tmp ;
		for(int k=0; k<descriptor_target.rows; k++){
			if(matches[i][k].distance <= mind)
				tmp.push_back(matches[i][k]) ;
			else
				break ;
		}
		if(!tmp.empty())
			good_matches.push_back(tmp) ;
	}
	//good_matches = matches ;
}

double ransac(vector<KeyPoint> &kps_obj, vector<KeyPoint> &kps_target, vector<Point2d> &pts_obj, vector<Point2d> &pts_target, vector<vector<DMatch>> &good_matches, Mat &homography){
	srand(time(NULL)) ;
	const int max_round = 80000 ;
	int count_round = 0 ;
	vector<Mat> candidates ;
	vector<int> candidates_inliers ;
	int max_inliers=0, max_index=-1 ;
	double rate=0.6 ;
	while(true){
		count_round++ ;
		if(count_round>200){
			rate *= 0.9 ;
			count_round = 0 ;
		}
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
						double error ;
						int num_inliers = inliers(pts_obj, pts_target, good_matches, transform_mat, 20.0, error) ;
						if(num_inliers > good_matches.size()*rate){
							homography = transform_mat.clone() ;
							cout << "inliers " << num_inliers << endl ;
							cout << "error " << error << endl ;
							return error ;

							//candidates.push_back(transform_mat.clone()) ;
							//candidates_inliers.push_back(num_inliers) ;
							//if(num_inliers > max_inliers){
							//	max_inliers = num_inliers ;
							//	max_index = candidates.size()-1 ;
							//}
						}
					}
				}
			}
		}
	}
}

void ransac_debug(vector<KeyPoint> &kps_obj, vector<KeyPoint> &kps_target, vector<Point2d> &pts_obj, vector<Point2d> &pts_target, vector<vector<DMatch>> &good_matches, Mat &homography, Mat img, Mat target){
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
						//cvWaitKey(0) ;
						double error ;
						int num_inliers = inliers(pts_obj, pts_target, good_matches, transform_mat, 5.0, error) ;
						if(num_inliers > good_matches.size()*0.8){
							cout << good_matches.size() << "\t" << num_inliers << "\t" << error << endl ;
							vector<Point2d> tmp, tmp2 ;
							tmp.push_back(pts_obj[good_matches[rndpts[0]][i].queryIdx]) ;
							tmp.push_back(pts_obj[good_matches[rndpts[1]][j].queryIdx]) ;
							tmp.push_back(pts_obj[good_matches[rndpts[2]][k].queryIdx]) ;
							tmp.push_back(pts_obj[good_matches[rndpts[3]][m].queryIdx]) ;
							perspectiveTransform(tmp, tmp2, transform_mat) ;
							cout << "Target Keypoints" << endl ;
							cout << pts_target[good_matches[rndpts[0]][i].trainIdx] << endl ;
							cout << pts_target[good_matches[rndpts[1]][j].trainIdx] << endl ;
							cout << pts_target[good_matches[rndpts[2]][k].trainIdx] << endl ;
							cout << pts_target[good_matches[rndpts[3]][m].trainIdx] << endl ;
							cout << "Object Keypoint & homography" << endl ;
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
							drawMatches(img, tmp_kps1, target, tmp_kps2, tmp_matches, img_matches) ;
							for(int a=0; a<4; a++)
								circle(img_matches, cvPoint((img.cols)+tmp2[a].x, tmp2[a].y), 5, Scalar(255, 0, 0), 2) ;
							imshow("matches", img_matches) ;
							vector<Point2d> obj_corner(4, Point2d(0, 0)), target_corner ;
							obj_corner[1] = Point2d(img.cols, 0) ;
							obj_corner[2] = Point2d(img.cols, img.rows) ;
							obj_corner[3] = Point2d(0, img.rows) ;
							perspectiveTransform(obj_corner, target_corner, transform_mat) ;
							Mat tmp_img = Mat(target).clone() ;
							for(int a=0; a<4; a++)
								circle(tmp_img, target_corner[a], 5, Scalar(255, 255, 0), 2) ;
							line(tmp_img, target_corner[0], target_corner[1], Scalar(0, 255, 0)) ;
							line(tmp_img, target_corner[1], target_corner[2], Scalar(0, 255, 0)) ;
							line(tmp_img, target_corner[2], target_corner[3], Scalar(0, 255, 0)) ;
							line(tmp_img, target_corner[3], target_corner[0], Scalar(0, 255, 0)) ;
							imshow("tmp", tmp_img) ;
							cvWaitKey(0) ;
							cvDestroyAllWindows() ;
						}
					}
				}
			}
		}
	}
}

void featureDetection(string obj, string target, double &error, Mat &result){
	Mat img_obj = imread(obj) ;
	Mat img_target = imread(target) ;
	SiftFeatureDetector detector(0.05, 5.0) ;
	vector<KeyPoint> kps_obj, kps_target ;
	vector<Point2d> pts_obj, pts_target ;
	detector.detect(img_obj, kps_obj) ;
	detector.detect(img_target, kps_target) ;
	keypoints2points(kps_obj, pts_obj) ;
	keypoints2points(kps_target, pts_target) ;
	cout << "keypoints: " << kps_obj.size() << "\t" << kps_target.size() << endl ;

	SiftDescriptorExtractor extractor ;
	Mat descriptor_obj, descriptor_target ;
	extractor.compute(img_obj, kps_obj, descriptor_obj) ;
	extractor.compute(img_target, kps_target, descriptor_target) ;

	vector<vector<DMatch>> matches(pts_obj.size()), good_matches ;
	knn_match(4, descriptor_obj, descriptor_target, matches, good_matches) ;
	cout << "good matches: " << good_matches.size() << endl ; ;

	Mat homography ;
	//ransac_debug(kps_obj, kps_target, pts_obj, pts_target, good_matches, homography, img_obj, img_target) ;
	error = ransac(kps_obj, kps_target, pts_obj, pts_target, good_matches, homography) ;
	cout << "niceHomography " << niceHomography(homography) << endl ;
	if(!niceHomography(homography))
		error = DBL_MAX ;
	//draw box
	vector<Point2d> obj_corner(4, Point2d(0, 0)), target_corner ;
	obj_corner[1] = Point2d(img_obj.cols, 0) ;
	obj_corner[2] = Point2d(img_obj.cols, img_obj.rows) ;
	obj_corner[3] = Point2d(0, img_obj.rows) ;
	perspectiveTransform(obj_corner, target_corner, homography) ;
	result = img_target.clone() ;
	for(int a=0; a<4; a++)
		circle(result, target_corner[a], 5, Scalar(255, 255, 0), 2) ;
	line(result, target_corner[0], target_corner[1], Scalar(0, 255, 0)) ;
	line(result, target_corner[1], target_corner[2], Scalar(0, 255, 0)) ;
	line(result, target_corner[2], target_corner[3], Scalar(0, 255, 0)) ;
	line(result, target_corner[3], target_corner[0], Scalar(0, 255, 0)) ;
	
}

int main(){
	Mat result ;
	double error = DBL_MAX ;
	vector<vector<string>> objs;
	string obj_a[] = {"object/a1.jpg", "object/a2.jpg", "object/a4.jpg", "object/a5.jpg", "object/a6.jpg"} ;
	string obj_b[] = {"object/b1.jpg", "object/b2.jpg", "object/b3.jpg"} ;
	string obj_c[] = {"object/c1.jpg", "object/c2.jpg", "object/c3.jpg", "object/c4.jpg"} ;
	objs.push_back(vector<string>(obj_a, obj_a + 5)) ;
	objs.push_back(vector<string>(obj_b, obj_b + 3)) ;
	objs.push_back(vector<string>(obj_c, obj_c + 4)) ;
	Mat tmp_mat ; 
	double tmp_error ;
	for(int i=0; i<objs[2].size(); i++){
		featureDetection(objs[2][i], "target/target1.jpg", tmp_error, tmp_mat) ;
		if(tmp_error < error){
			error = tmp_error ;
			result = tmp_mat.clone() ;
		}
		string name = "a" ;
		name[0] += i ;
		imshow(name, tmp_mat) ;
		cout << i << "\t" << tmp_error << endl ;
		cout << "------------------------" << endl ;
	}
	imshow("result", result) ;
	waitKey(0) ;
	return 0 ;
}