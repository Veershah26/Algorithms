#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define NBUCKETS 5  /* Number of buckets */

struct Node {
    int data;
    struct Node *next;
};

/* Function prototypes */
void print_array(const int *array, int n);
void bucket_sort(int *array, int n);
struct Node *insertion_sort(struct Node *list);
int get_bucket_index(int value, int min_value, int max_value);
int *load_from_file(const char *filename, int *array_size);
void test();

/* Main function */
int main(int argc, const char *argv[]) {
    srand(time(NULL));
    test();
    return 0;
}

/* Function to print the array */
void print_array(const int *array, int n) {
    for(int i = 0; i < n; i++) {
        printf("%d ", array[i]);
    }
    printf("\n");
}

/* Function to determine the bucket index for a value */
int get_bucket_index(int value, int min_value, int max_value) {
    int range = max_value - min_value + 1;
    int bucket_size = range / NBUCKETS;
    if (bucket_size == 0) bucket_size = 1;  // Ensure non-zero bucket size
    return (value - min_value) / bucket_size;
}

/* Insertion sort for sorting elements within a bucket */
struct Node *insertion_sort(struct Node *list) {
    struct Node *sorted_list = NULL;

    while (list != NULL) {
        struct Node *current = list;
        list = list->next;

        if (sorted_list == NULL || sorted_list->data >= current->data) {
            current->next = sorted_list;
            sorted_list = current;
        } else {
            struct Node *temp = sorted_list;
            while (temp->next != NULL && temp->next->data < current->data) {
                temp = temp->next;
            }
            current->next = temp->next;
            temp->next = current;
        }
    }

    return sorted_list;
}

/* Bucket sort implementation */
void bucket_sort(int *array, int n) {
    int min_value = array[0], max_value = array[0];

    /* Find the minimum and maximum values in the array */
    for (int i = 1; i < n; i++) {
        if (array[i] < min_value) min_value = array[i];
        if (array[i] > max_value) max_value = array[i];
    }

    struct Node **buckets = (struct Node **)malloc(sizeof(struct Node *) * NBUCKETS);
    if (buckets == NULL) {
        printf("Memory Allocation Failed for Buckets!\n");
        return;
    }

    /* Initialize the buckets */
    for (int i = 0; i < NBUCKETS; ++i) {
        buckets[i] = NULL;
    }

    /* Distribute the array elements into the appropriate buckets */
    for (int i = 0; i < n; ++i) {
        int bucket_index = get_bucket_index(array[i], min_value, max_value);

        // Ensure bucket_index is within the range of buckets
        if (bucket_index >= NBUCKETS) {
            bucket_index = NBUCKETS - 1;
        }

        struct Node *new_node = (struct Node *)malloc(sizeof(struct Node));
        if (new_node == NULL) {
            printf("Memory Allocation Failed for Bucket Node!\n");
            free(buckets);
            return;
        }
        new_node->data = array[i];
        new_node->next = buckets[bucket_index];
        buckets[bucket_index] = new_node;
    }

    /* Sort each bucket and gather them back into the array */
    int index = 0;
    for (int i = 0; i < NBUCKETS; ++i) {
        buckets[i] = insertion_sort(buckets[i]);
        struct Node *node = buckets[i];
        while (node != NULL) {
            array[index++] = node->data;
            struct Node *temp = node;
            node = node->next;
            free(temp);
        }
    }

    /* Free the buckets array */
    free(buckets);
}

/* Function to load an array from a file */
int *load_from_file(const char *filename, int *array_size) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        printf("Failed To Open File %s\n", filename);
        return NULL;
    }

    *array_size = 0;
    int temp;
    while (fscanf(file, "%d", &temp) == 1) {
        (*array_size)++;
    }

    int *array = (int *)malloc((*array_size) * sizeof(int));
    if (array == NULL) {
        printf("Memory Allocation Failed!\n");
        fclose(file);
        return NULL;
    }

    rewind(file);
    for (int i = 0; i < *array_size; i++) {
        if (fscanf(file, "%d", &array[i]) != 1) {
            printf("Failed to read element %d\n", i);
            free(array);
            fclose(file);
            return NULL;
        }
    }

    fclose(file);
    return array;
}

/* Test function */
void test() {
    int array_size;
    int *array = NULL;

    printf("Enter 1 To Input Manually, 2 To Input From File: ");
    int choice;
    scanf("%d", &choice);

    if (choice == 1) {
        printf("Enter Array Size: ");
        scanf("%d", &array_size);

        array = (int *)malloc(array_size * sizeof(int));
        if (array == NULL) {
            printf("Memory Allocation Failed!\n");
            return;
        }

        printf("Enter Array Elements:\n");
        for (int i = 0; i < array_size; i++) {
            scanf("%d", &array[i]);
        }
    } else if (choice == 2) {
        char filename[100];
        printf("Enter Filename: ");
        scanf("%s", filename);

        array = load_from_file(filename, &array_size);
        if (array == NULL) {
            return;
        }
    } else {
        printf("Invalid Choice!\n");
        return;
    }

    printf("Original Array:\n");
    print_array(array, array_size);

    clock_t start = clock();
    bucket_sort(array, array_size);
    clock_t end = clock();

    printf("Sorted Array:\n");
    print_array(array, array_size);

    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Time Taken: %0.6f Seconds\n", time_taken);

    free(array);
}
