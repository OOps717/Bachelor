#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <sys/wait.h>

void error (int er_num)
{   
    if (er_num == 1)
    {
        fprintf (stderr,"Input should be: detect [-t format] [-i interval] [-l limit] [-c] prog arg ... arg\n");
        exit (1);
    }
    else if (er_num == 2)
	{
        fprintf (stderr,"Time format is too big\n");
        exit (2);
    }
	else if (er_num == 3) 
	{
		fprintf (stderr,"Not appropriate time interval. Can not be negative or NULL interval\n");
		exit (3);
	}
	else if (er_num == 4)
	{
		fprintf (stderr,"Not appropriate time limit. Can not be negative limit\n");
		exit (4);
	}
	else if (er_num == 5)
	{
		fprintf (stderr, "Expected argument after options\n");
        exit (5);
	}
	else if (er_num == 6)
	{
		fprintf (stderr,"Pipe is not created\n");
		exit(6);
	}
	else if (er_num == 7)
	{
		fprintf (stderr,"Error in fork\n");
		exit(7);
	}
	else if (er_num == 9)
	{
		fprintf (stderr, "Error in reallocation\n");
		exit (9);
	}
	
}

void init_time (const char *format)  //showing current time with following format
{
    char *time_buf = (char*) malloc(70 * sizeof(char));
	
	const time_t t_start = time (NULL) ;
	struct tm *tm = localtime (&t_start) ;
	
	if (strftime (time_buf, 70, format, tm) == 0) error(2);
	printf ("%s\n", time_buf) ;

    free(time_buf);
}


int main (int argc, char *argv [])
{
    if (argc == 1)	error (1);

    int p[2];
	char* format = NULL;
    int interval = 10000 * 1000;
    int limit = 0;
    char returnCode = 0;
    int option = 0;
    char* output = (char*) calloc(20, sizeof(char));  //program output after all limits and intervals
    int prevCode = 210121;  //to check if the program worked in previous time of launch after first launch 

	//Checking and getting options ----------------------------------------------------
    while ((option = getopt(argc, argv, "+t:i:l:c")) != -1)
    {
        switch (option) 
        {
            case 'c':
                returnCode = 1;
                break;
            case 'i':
                interval = atoi(optarg);
                interval *= 1000;
				if (interval <= 0) error (3);
                break;
            case 'l':
                limit = atoi(optarg);
				if (limit < 0) error (4);
                break;
            case 't':
                format = optarg;
                break;
			default :
				error (1);
        }
    }
	//------------------------------------------------------------------

	if (optind >= argc) error (5); //if there is no program

    if (limit == 0) limit=1;

	while (limit) 
    {
        if (pipe(p) == -1) error(6);
        
        switch (fork())
        {
            case -1:
                error(7);
                break;
            case 0:
                close(p[0]);
                dup2(p[1], 1);
                close(p[1]);
                execvp(argv [optind], argv + optind);//executing program
                exit(8);
                break;
            default:
                close(p[1]);
                dup2(p[0], 0);
                close(p[0]);                                                                                                                                                              
			
                char* courant;
                courant = (char*) calloc(20, sizeof(char));  //our buffer output
                double taille = 20;  //size of buffer 
                size_t deja_lit = 0;  //size of already read characters
                double avant = deja_lit;  //number of previously read characters
    
                deja_lit += read(0, courant + deja_lit, taille - deja_lit - 1);  //reading the given number of bites while saving it in buffer
                if (deja_lit >= taille - 1) //if read bytes is more than size of buffer it will expand it
                {
                    taille*=2;
                    courant = (char*) realloc (courant, taille);
                    if (!courant) error(9);
                }

                while (avant != deja_lit)  //looping previous lines ↑↑↑
                {
                    avant = deja_lit;
                    deja_lit += read(0, courant + deja_lit, taille - deja_lit - 1);
                    if (deja_lit >= taille - 1) {
                        taille*=2;
                        courant = (char*) realloc (courant, taille);
                        if (!courant) error(9);
                    }
                }

			    if (memcmp(courant,output,taille))  //to compare if current buffer is similar to the output or not
                {
                    write(1, courant, deja_lit);  //in case of discrepancy with previous output puts current buf to input 
                    output = courant;
                }
            
            
                int check;
                wait(&check);

                if (returnCode && (prevCode != WEXITSTATUS(check) || prevCode == 210121)) //check if it was launched previously and the previous code is not the current one
                {
                    prevCode = WEXITSTATUS(check);
                    printf("exit %d\n", prevCode);
                    fflush(0);  //to flush output buffer into output of the program
                }

                if (format != NULL) init_time(format);
                usleep(interval);
        }
        limit--;
    }
    
    return 0;
}


