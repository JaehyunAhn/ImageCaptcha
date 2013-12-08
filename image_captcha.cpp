 /*
  * Author: Sogo
  * sogosonnet@gmail.com
  * Contents: Image Captcha
  * Date: 2013/12/7 ~
  * input: keyboard (directory)
  * output: all files in directory
  *
  *
  * <Open Source provider>
  * Author: Liu Liu
  * Contents: SURF Image Comparison Algorithm
  * liuliu.1987+opencv@gmail.com
  */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <dirent.h>

#include <cv.h>
#include <highgui.h>
#include <ctype.h>

#include <iostream>
#include <vector>

using namespace std;

// define whether to use approximate nearest-neighbor search
#define USE_FLANN

IplImage *image = 0;

double
compareSURFDescriptors( const float *d1, const float *d2, 
                        double best, int length)
{
    double total_cost = 0;
    assert( length % 4 == 0 );
    for( int i = 0; i < length; i+= 4 )
    {
        double t0 = d1[i] - d2[i];
        double t1 = d1[i+1] - d2[i+1];
        double t2 = d1[i+2] - d2[i+2];
        double t3 = d1[i+3] - d2[i+3];
        total_cost += t0*t0 + t1*t1 + t2*t2 + t3*t3;
        if( total_cost > best )
            break;
    }
    return total_cost;
}// 42

int
naiveNearestNeighbor( const float* vec, int laplacian,
                      const CvSeq* model_keypoints,
                      const CvSeq* model_descriptors )
{
    int length = (int)(model_descriptors->elem_size/sizeof(float));
    int i, neighbor = -1;
    double d, dist1 = 1e6, dist2 = 1e6;
    CvSeqReader reader, kreader;
    cvStartReadSeq( model_keypoints, &kreader, 0 );
    cvStartReadSeq( model_descriptors, &reader, 0 );

    for( i = 0; i < model_descriptors->total; i++ )
    {
        const CvSURFPoint* kp = (const CvSURFPoint*)kreader.ptr;
        const float* mvec = (const float*)reader.ptr;
        CV_NEXT_SEQ_ELEM( kreader.seq->elem_size, kreader );
        CV_NEXT_SEQ_ELEM( reader.seq->elem_size, reader );
        if( laplacian != kp->laplacian )
            continue;
        d = compareSURFDescriptors( vec, mvec, dist2, length );
        if( d < dist1 )
        {
            dist2 = dist1;
            dist1 = d;
            neighbor = i;
        }
        else if ( d < dist2 )
            dist2 = d;
    }
    if ( dist1 < 0.6*dist2 )
        return neighbor;
    return -1;
}// 79

void
findPairs( const CvSeq* objectKeypoints, const CvSeq* objectDescriptors,
           const CvSeq* imageKeypoints, const CvSeq* imageDescriptors, 
           vector<int>& ptpairs )
{
    int i;
    CvSeqReader reader, kreader;
    cvStartReadSeq( objectKeypoints, &kreader );
    cvStartReadSeq( objectDescriptors, &reader );
    ptpairs.clear();

    for( i = 0; i < objectDescriptors->total; i++ )
    {
        const CvSURFPoint* kp = (const CvSURFPoint*)kreader.ptr;
        const float* descriptor = (const float*)reader.ptr;
        CV_NEXT_SEQ_ELEM( kreader.seq->elem_size, kreader );
        CV_NEXT_SEQ_ELEM( reader.seq->elem_size, reader );
        int nearest_neighbor = naiveNearestNeighbor (descriptor, kp->laplacian, 
                                                     imageKeypoints, imageDescriptors );
        if( nearest_neighbor >= 0 )
        {
            ptpairs.push_back(i);
            ptpairs.push_back(nearest_neighbor);
        }
    }
}// 103

void
flannFindPairs( const CvSeq*, const CvSeq* objectDescriptors,
           const CvSeq*, const CvSeq* imageDescriptors, vector<int>& ptpairs )
{
    int length = (int)(objectDescriptors->elem_size/sizeof(float));
    cv::Mat m_object(objectDescriptors->total, length, CV_32F);
    cv::Mat m_image(imageDescriptors->total, length, CV_32F);

    // copy descriptors
    CvSeqReader obj_reader;
    float* obj_ptr = m_object.ptr<float>(0);
    cvStartReadSeq( objectDescriptors, &obj_reader );
    for(int i = 0; i < objectDescriptors->total; i++ )
    {
        const float* descriptor = (const float*)obj_reader.ptr;
        CV_NEXT_SEQ_ELEM( obj_reader.seq->elem_size, obj_reader );
        memcpy(obj_ptr, descriptor, length*sizeof(float));
        obj_ptr += length;
    }
    CvSeqReader img_reader;
    float* img_ptr = m_image.ptr<float>(0);
    cvStartReadSeq( imageDescriptors, &img_reader );
    for(int i = 0; i < imageDescriptors->total; i++ )
    {
        const float* descriptor = (const float*)img_reader.ptr;
        CV_NEXT_SEQ_ELEM( img_reader.seq->elem_size, img_reader );
        memcpy(img_ptr, descriptor, length*sizeof(float));
        img_ptr += length;
    }

    // find nearest neighbors using FLANN
    cv::Mat m_indices(objectDescriptors->total, 2, CV_32S);
    cv::Mat m_dists(objectDescriptors->total, 2, CV_32F);
    cv::flann::Index flann_index(m_image, cv::flann::KDTreeIndexParams(4)); // using 4 randomized kdtrees
    flann_index.knnSearch(m_object, m_indices, m_dists, 2, cv::flann::SearchParams(64) ); // maximum number of leafs checked

    int* indices_ptr = m_indices.ptr<int>(0);
    float* dists_ptr = m_dists.ptr<float>(0);
    for (int i=0;i<m_indices.rows;++i) {
        if (dists_ptr[2*i]<0.6*dists_ptr[2*i+1]) {
            ptpairs.push_back(i);
            ptpairs.push_back(indices_ptr[2*i]);
        }
    }
} // 152


/* a rough implementation for object location */
int
locatePlanarObject( const CvSeq* objectKeypoints, const CvSeq* objectDescriptors,
                    const CvSeq* imageKeypoints, const CvSeq* imageDescriptors,
                    const CvPoint src_corners[4], CvPoint dst_corners[4] )
{
    double h[9];
    CvMat _h = cvMat(3, 3, CV_64F, h);
    vector<int> ptpairs;
    vector<CvPoint2D32f> pt1, pt2;
    CvMat _pt1, _pt2;
    int i, n;

#ifdef USE_FLANN
    flannFindPairs( objectKeypoints, objectDescriptors, imageKeypoints, imageDescriptors, ptpairs );
#else
    findPairs( objectKeypoints, objectDescriptors, imageKeypoints, imageDescriptors, ptpairs );
#endif

    n = ptpairs.size()/2;
    if( n < 4 )
        return 0;

    pt1.resize(n);
    pt2.resize(n);
    for( i = 0; i < n; i++ )
    {
        pt1[i] = ((CvSURFPoint*)cvGetSeqElem(objectKeypoints,ptpairs[i*2]))->pt;
        pt2[i] = ((CvSURFPoint*)cvGetSeqElem(imageKeypoints,ptpairs[i*2+1]))->pt;
    }

    _pt1 = cvMat(1, n, CV_32FC2, &pt1[0] );
    _pt2 = cvMat(1, n, CV_32FC2, &pt2[0] );
    if( !cvFindHomography( &_pt1, &_pt2, &_h, CV_RANSAC, 5 ))
        return 0;

    for( i = 0; i < 4; i++ )
    {
        double x = src_corners[i].x, y = src_corners[i].y;
        double Z = 1./(h[6]*x + h[7]*y + h[8]);
        double X = (h[0]*x + h[1]*y + h[2])*Z;
        double Y = (h[3]*x + h[4]*y + h[5])*Z;
        dst_corners[i] = cvPoint(cvRound(X), cvRound(Y));
    }

    return 1;
} // 201


int 
folder_thriver(char **repo, int *repo_count, int *repo_thrive);

void 
description();

int main (int arc, char **argv) {
    /* input exception handler */
    char current[2];
    if(arc == 1) 
        return 0;
    
    current[0] = argv[1][0];
    current[1] = '\0';

    if(strcmp(current,".") != 0)
        return 0;

    /* Description */
    description();
    
    /* folder thrive */
    char **dir_route;
    int i;
    int dir_count = 0;
    int repo_count = 0;
    
    dir_route = (char **)malloc(sizeof(char*)*1000);
    for(i=0; i<1000; i++)
        dir_route[i] = (char *)malloc(sizeof(char*)*2000);

    strcpy(dir_route[dir_count],argv[1]);

    while(1) {
        if(repo_count != 0)
            if(dir_count == repo_count)
                break;
        if(folder_thriver(dir_route, &dir_count, &repo_count) == 0)
            break;
        if(dir_count == 0)
            break;
    }

    for(i=0; i<1000; i++)
        free(dir_route[i]);
    free(dir_route);

    return 1;
}

/****************************
  * input:  1. repo(sitory) array
  *         2. repo count
  *          
  * output: 1. files in repository
  *         2. save other directory
  * return: == 1 : everything goes correct
            == 0 : return NULL
  *
  ***************************/
int folder_thriver(char **repo, int *repo_count, int *repo_thrive) {
    // printf("Current route:%s\n",repo[*repo_count]);

    int file_counter = 0;
    int init = *repo_count;
    int count = *repo_thrive;
    DIR *dirp;
    struct dirent *entry;
    char init_dir[10000];

    strcpy(init_dir,repo[init]);

    dirp = opendir(init_dir);

    if(dirp) {
        while (( entry = readdir(dirp)) != NULL ) {
            if( entry->d_type == DT_REG ) {
                if( file_counter < 1 )
                {
                    if ( strstr(entry->d_name,".jpg") )
                        strcpy(init_dir,entry->d_name);
                        //printf("\t\t%s\n",entry->d_name);
                    if ( strstr(entry->d_name,".jpeg") )
                        strcpy(init_dir,entry->d_name);
                        //printf("\t\t%s\n",entry->d_name);
                    if ( strstr(entry->d_name,".png") )
                        strcpy(init_dir,entry->d_name);
                        //printf("\t\t%s\n",entry->d_name);
                }
                file_counter++;
            }
            if( entry->d_type == DT_DIR ) {
                if(entry->d_name[0] != '.') {
                    if(count < 1000) {
                        strcpy(repo[count+1],init_dir);
                        strcat(repo[count+1],"/");
                        strcat(repo[count+1],entry->d_name);
                        //printf("%s\n",repo[count+1]);
                        count++;
                    }
                    else {
                        printf("ERROR: [file_triver.c] Repository Count over 1000.\n");
                        return 0;
                    }
                }
            }
        }
        if( file_counter >= 1 )
        {
            printf("Current route:%s\n",repo[*repo_count]);
            printf("\t\thas %s\n",init_dir);
        }
        closedir(dirp);
    }
    else {
        printf("ERROR: [file_triver.c] There is no directory: %s\n",init_dir);
        return 0;
    }
    *repo_thrive = count;
    *repo_count+=1;
    return 1;
}

void description () {
    printf("That folder has:\n");
}
