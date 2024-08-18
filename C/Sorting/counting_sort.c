#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void print_array(const int *array, int n) {
    for (int i = 0; i < n; ++i) {
        printf("%d ", array[i]);
    }
    printf("\n");
}

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
        fscanf(file, "%d", &array[i]);
    }

    fclose(file);
    return array;
}

void counting_sort(int *array, int size) {
    int max_val = 0;
    for (int i = 0; i < size; i++) {
        if (array[i] > max_val) {
            max_val = array[i];
        }
    }

    int *count = (int *)malloc((max_val + 1) * sizeof(int));
    memset(count, 0, (max_val + 1) * sizeof(int));

    for (int i = 0; i < size; i++) {
        count[array[i]]++;
    }

    printf("Sorted array:\n");
    for (int i = 0; i <= max_val; i++) {
        while (count[i] > 0) {
            printf("%d ", i);
            count[i]--;
        }
    }
    printf("\n");

    free(count);
}

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
    counting_sort(array, array_size);
    clock_t end = clock();

    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Time Taken: %0.6f Seconds\n", time_taken);

    free(array);
}

int main() {
    srand(time(NULL));
    test();
    return 0;
}
