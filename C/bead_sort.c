#include <stdio.h>
#include <stdlib.h>

#define BEAD(i, j) beads[i * maximum + j]

void display(const int *array, int x) {
    for (int i = 0; i < x; i++) {
        printf("%d ", array[i]);
    }
    printf("\n");
}

void bead_sort(int *a, size_t length) {
    int i, j, maximum, sum;
    unsigned char *beads;

    for (i = 1, maximum = a[0]; i < length; i++)
        if (a[i] > maximum)
            maximum = a[i];

    beads = calloc(1, maximum * length);

    for (i = 0; i < length; i++) {
        for (j = 0; j < a[i]; j++) BEAD(i, j) = 1;
    }


    for (j = 0; j < maximum; j++) {
        for (sum = i = 0; i < length; i++) {
            sum += BEAD(i, j);
            BEAD(i, j) = 0;
        }
        for (i = length - sum; i < length; i++) BEAD(i, j) = 1;
    }

    for (i = 0; i < length; i++) {
        for (j = 0; j < maximum && BEAD(i, j); j++);
        a[i] = j;
    }
    free(beads);
}

int main(int argc, const char *argv[]) {
    int x;
    printf("Enter Size Of Array: ");
    scanf("%d", &x);  

    printf("Enter Elements: \n");
    int i;
    int *array = (int *)malloc(x * sizeof(int));
    for (i = 0; i < x; i++) {
        scanf("%d", &array[i]);
    }

    printf("Original Array: ");
    display(array, x);

    bead_sort(array, x);

    printf("Sorted Array: ");
    display(array, x);

    free(array);
    return 0;
}
