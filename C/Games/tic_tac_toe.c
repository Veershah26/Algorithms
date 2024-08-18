#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void single_player();
void multi_player();
void place_x(int);  
void place_o(int); 
int check_win();    
void display_board(); 

char game_board[9];

int main() {
    srand((unsigned int)time(NULL));
    int play_again;

    do {
        int choice;
        for (int i = 0; i < 9; i++) game_board[i] = '*';

        printf("***************************************\n");
        printf("************* TIC-TAC-TOE *************\n");
        printf("***************************************\n");
        printf("1. Single Player (You vs Computer)\n");
        printf("2. Multiplayer (You vs Player)\n");
        printf("3. Exit\n");
        printf("Enter Your Choice: ");
        scanf("%d", &choice);

        switch (choice) {
            case 1:
                single_player();
                break;
            case 2:
                multi_player();
                break;
            case 3:
                printf("Thank You For Playing! Exiting...\n");
                return 0;
            default:
                printf("Invalid Choice. Please Try Again.\n");
        }

        printf("Play Again? Enter 1 For YES or 0 For NO: ");
        scanf("%d", &play_again);

    } while (play_again == 1);

    return 0;
}

void single_player() {
    int position;
    int move_count = 0;

    display_board();

    for (int turn = 0; turn < 9; turn++) {
        printf("Where Would You Like To Place 'X'? ");
        scanf("%d", &position);

        place_x(position);
        move_count++;
        display_board();

        if (move_count < 5) {
            place_o(rand() % 9 + 1);
            display_board();
        }

        int result = check_win();
        if (result == -1) {
            printf("Congratulations! You Win!\n");
            return;
        } else if (result == -2) {
            printf("Sorry, You Lose! Computer Wins.\n");
            return;
        }

        if (move_count == 5) {
            printf("It's A Draw!\n");
            return;
        }
    }
}

void multi_player() {
    int position;
    int move_count = 0;

    display_board();

    for (int turn = 0; turn < 9; turn++) {
        if (turn % 2 == 0) {
            printf("Player 1 - Where Would You Like To Place 'X'? ");
            scanf("%d", &position);
            place_x(position);
        } else {
            printf("Player 2 - Where Would You Like To Place 'O'? ");
            scanf("%d", &position);
            place_o(position);
        }

        move_count++;
        display_board();

        int result = check_win();
        if (result == -1) {
            printf("Player 1 Wins!\n");
            return;
        } else if (result == -2) {
            printf("Player 2 Wins!\n");
            return;
        }

        if (move_count == 9) {
            printf("It's A Draw!\n");
            return;
        }
    }
}

void place_x(int position) {
    while (position < 1 || position > 9 || game_board[position - 1] != '*') {
        printf("Invalid Move. Please Enter A Number Between 1 & 9: ");
        scanf("%d", &position);
    }
    game_board[position - 1] = 'X';
}

void place_o(int position) {
    while (position < 1 || position > 9 || game_board[position - 1] != '*') {
        position = rand() % 9 + 1;
    }
    game_board[position - 1] = 'O';
    printf("Computer Placed 'O' At Position %d\n", position);
}

int check_win() {
    int winPatterns[8][3] = {
        {0, 1, 2}, {3, 4, 5}, {6, 7, 8}, 
        {0, 3, 6}, {1, 4, 7}, {2, 5, 8}, 
        {0, 4, 8}, {2, 4, 6}             
    };

    for (int i = 0; i < 8; i++) {
        int a = winPatterns[i][0];
        int b = winPatterns[i][1];
        int c = winPatterns[i][2];

        if (game_board[a] == game_board[b] && game_board[b] == game_board[c]) {
            if (game_board[a] == 'X') return -1;
            if (game_board[a] == 'O') return -2;
        }
    }
    return 0;
}


void display_board() {
    printf("\n");
    for (int i = 0; i < 9; i++) {
        printf("%c ", game_board[i]);
        if ((i + 1) % 3 == 0) printf("\n");
    }
    printf("\n");
}
