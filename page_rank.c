#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>

typedef struct Node{
    int in_node;
    double wieght;
    struct Node *next;
}List;

List* insertnode(List* last_node, int new_in_node, int new_wieght)
{
        List *head=last_node;
        List *new_node = (List *)malloc(sizeof(List));
        new_node->in_node = new_in_node;
        new_node->wieght = new_wieght;
        new_node->next = NULL;
        if (last_node==NULL)
        {
                return new_node;
        }
        while (last_node->next != NULL)
        {
                last_node = last_node->next;
        }
        last_node->next=new_node;
        return head;
}

void printlist(List* temp_node)
{
    while(temp_node!=NULL)
    {
        printf("Node = %d, wieght =%lf\n",temp_node->in_node,temp_node->wieght);
        temp_node=temp_node->next;
    }
}


int main(int argc, char **argv)
{
        int mat_size = 50000,rank,size,*total_out;
        double *pr_vect_old, *pr_vect_new, *pr_vect_gather,error;
        MPI_Init(&argc, &argv);
        MPI_Status status;
        MPI_Request request;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        mat_size = (mat_size%size==0) ? mat_size : ((mat_size/size + 1)*size);
        //List array for saving all incomming nodes to a particular node distributed among processors
        List **linked_list =  (List**) malloc ( mat_size/size * sizeof(List*));
        for (int z = 0; z < mat_size/size; z++)
        {
                linked_list[z] = NULL;
        }
        //Array for saving total out wieght from a node
        total_out = (int *)calloc(mat_size,sizeof(int));
        //Reading of file to include all the incomming nodes in a list of a node
        FILE *fp;
        char *myText = NULL;
        size_t len = 0;
        ssize_t read;
        fp = fopen("graph.dat", "r");
        while ((read = getline(&myText, &len, fp)) != -1)
        {
                if (myText[0] == 'a')
                {
                        int nodes[3];
                        int j = 0;
                        for (int i = 0; i < len; i++)
                        {
                                if(j==3)
                                {break;}
                                if (myText[i] >= '0' && myText[i] <= '9')
                                {
                                        int num = 0;
                                        while (i < len && myText[i] >= '0' && myText[i] <= '9')
                                        {
                                                int y = (int)myText[i] - '0';
                                                num = num * 10 + y;
                                                i++;
                                        }
                                        nodes[j] = num;
                                        j++;
                                }
                        }
                        //Calculating total out wieght of each node
                        if (rank==0)
                        {
                                total_out[nodes[0]]+=nodes[2];
                        }
                        int upbound = (rank + 1) * mat_size / size;
                        int lwbound = rank * mat_size / size;
                        //Loading the lists array on the differnt processors
                        if ( nodes[1]>=lwbound && nodes[1] < upbound)
                        {
                                //printf("proc = %d , %d, %d, %d\n",rank,nodes[0],nodes[1],nodes[2]);
                                int index1 = nodes[1]-rank*mat_size/size;
                                int index2 = nodes[0];
                                linked_list[index1]=insertnode(linked_list[index1],index2,nodes[2]);
                        }
                }
        }
        MPI_Bcast(total_out, mat_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        double start_time = MPI_Wtime();
        //Saving the out probablities of each edges from a node in list wiegth 
        int qi=0;
        while (qi<mat_size/size)
        {
                List *iter = linked_list[qi];
                while (iter!=NULL)
                {
                        if (total_out[iter->in_node] != 0)
                        {
                                iter->wieght = iter->wieght / total_out[iter->in_node];
                        }
                        iter=iter->next;
                }
                qi++;
        }
        //Arrays for maintaining pageranks
        pr_vect_new = (double *)malloc(mat_size/size * sizeof(double));
        pr_vect_old = (double *)malloc(mat_size/size * sizeof(double));
        //For making available every processor previous page-rank of in nodes
        pr_vect_gather = (double *)malloc(mat_size * sizeof(double));
        //Initialization of arrays
        if (rank == 0)
        {
                for (int z = 0; z < mat_size; z++)
                {
                        pr_vect_gather[z] = 1 / mat_size;
                }
        }
        for (int z = 0; z < mat_size/size; z++)
        {
                pr_vect_old[z] = 1 / mat_size;
        }
        //Loop for iterations
        double d=0.85;
        int qr=0;
        while(qr<50)
        {
                MPI_Bcast(pr_vect_gather, mat_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                int ql=0;
                double sum =0;
                while (ql < mat_size / size)
                {
                        //Using pagerank formula
                        pr_vect_new[ql] =(1-d)/mat_size;
                        List *iter = linked_list[ql];
                        while (iter != NULL)
                        {
                                pr_vect_new[ql] += d * pr_vect_gather[iter->in_node] * iter->wieght;
                                iter = iter->next;
                        }
                        //For calculation of error
                        sum += (pr_vect_new[ql]-pr_vect_old[ql])*(pr_vect_new[ql]-pr_vect_old[ql]);
                        pr_vect_old[ql]=pr_vect_new[ql];
                        ql++;
                }
                MPI_Gather(pr_vect_new, mat_size / size, MPI_DOUBLE, pr_vect_gather, mat_size / size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                MPI_Reduce(&sum, &error, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
                MPI_Bcast(&error, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                error=sqrt(error);
                //Applying tolerance condition
                if(error<pow(10,-7))
                {break;}
                qr++;
        }
        
        double end_time= MPI_Wtime()-start_time;

        double exec_time;
        MPI_Reduce(&end_time, &exec_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        //Printing of results
        if (rank == 0)
        {
                printf("Iteration = %d\n",qr);
                printf("time for %d processors = %lf sec\n",size,exec_time/size);
                for (int z = 0; z < mat_size; z++)
                {
                        printf("%d , %lf\n",z,pr_vect_gather[z]);
                }
        }
        MPI_Finalize();
}