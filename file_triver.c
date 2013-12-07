/*****************************
  * sogo
  * 2013/12/5 ~ 2013/12/7
  * file triver ver 2
  * input: keyboard (directory)
  * output: all files in directory
  ****************************/
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <dirent.h>

int folder_thriver(char **repo, int *repo_count, int *repo_thrive);
void description();

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
    printf("Current route:%s\n",repo[*repo_count]);

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
                if ( strstr(entry->d_name,".jpg") )
                    printf("\t\t%s\n",entry->d_name);
                if ( strstr(entry->d_name,".jpeg") )
                    printf("\t\t%s\n",entry->d_name);
                if ( strstr(entry->d_name,".png") )
                    printf("\t\t%s\n",entry->d_name);
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
