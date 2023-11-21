#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_RULES 10
#define MAX_TERMINALS 10
#define MAX_NON_TERMINALS 10

char nonTerminals[MAX_NON_TERMINALS];
char terminals[MAX_TERMINALS];
char parseTable[MAX_NON_TERMINALS][MAX_TERMINALS][MAX_RULES];
char productions[MAX_NON_TERMINALS][MAX_RULES][100];

int numNonTerminals, numTerminals;

int isTerminal(char symbol) {
    for (int i = 0; i < numTerminals; i++) {
        if (terminals[i] == symbol) {
            return 1;
        }
    }
    return 0;
}

int findNonTerminalIndex(char symbol) {
    for (int i = 0; i < numNonTerminals; i++) {
        if (nonTerminals[i] == symbol) {
            return i;
        }
    }
    return -1;
}

void computeFirst(char symbol, int ntIndex) {
    for (int i = 0; i < MAX_RULES && productions[ntIndex][i][0] != '\0'; i++) {
        char firstSymbol = productions[ntIndex][i][0];
        if (isTerminal(firstSymbol)) {
            parseTable[ntIndex][firstSymbol][i] = 1;
        } else {
            int firstIndex = findNonTerminalIndex(firstSymbol);
            if (firstIndex != -1) {
                computeFirst(firstSymbol, firstIndex);
                for (int j = 0; j < numTerminals; j++) {
                    if (parseTable[firstIndex][terminals[j]][0] == 1) {
                        parseTable[ntIndex][terminals[j]][i] = 1;
                    }
                }
            }
        }
    }
}

void computeFollow(char symbol, int ntIndex) {
    for (int i = 0; i < numNonTerminals; i++) {
        for (int j = 0; j < MAX_RULES && productions[i][j][0] != '\0'; j++) {
            for (int k = 0; k < strlen(productions[i][j]); k++) {
                if (productions[i][j][k] == symbol) {
                    if (k < strlen(productions[i][j]) - 1) {
                        char followSymbol = productions[i][j][k + 1];
                        if (isTerminal(followSymbol)) {
                            parseTable[i][followSymbol][j] = 1;
                        } else {
                            int followIndex = findNonTerminalIndex(followSymbol);
                            if (followIndex != -1) {
                                computeFirst(followSymbol, followIndex);
                                for (int l = 0; l < numTerminals; l++) {
                                    if (parseTable[followIndex][terminals[l]][0] == 1) {
                                        parseTable[i][terminals[l]][j] = 1;
                                    }
                                }
                            }
                        }
                    } else {
                        if (i != ntIndex) {
                            computeFollow(nonTerminals[i], i);
                        }
                    }
                }
            }
        }
    }
}

int main() {
    printf("Enter the number of non-terminals: ");
    scanf("%d", &numNonTerminals);
    printf("Enter the non-terminals: ");
    for (int i = 0; i < numNonTerminals; i++) {
        scanf(" %c", &nonTerminals[i]);
    }

    printf("Enter the number of terminals: ");
    scanf("%d", &numTerminals);
    printf("Enter the terminals: ");
    for (int i = 0; i < numTerminals; i++) {
        scanf(" %c", &terminals[i]);
    }

    for (int i = 0; i < numNonTerminals; i++) {
        for (int j = 0; j < numTerminals; j++) {
            for (int k = 0; k < MAX_RULES; k++) {
                parseTable[i][terminals[j]][k] = 0;
            }
        }
    }

    for (int i = 0; i < numNonTerminals; i++) {
        printf("Enter the productions for %c (use '|' to separate rules, end with '$'): ", nonTerminals[i]);
        int j = 0;
        while (1) {
            char c;
            scanf(" %c", &c);
            if (c == '$') {
                productions[i][j][0] = '\0';
                break;
            }
            if (c == '|') {
                productions[i][j][0] = '\0';
                j++;
            } else {
                productions[i][j][strlen(productions[i][j])] = c;
            }
        }
    }

    for (int i = 0; i < numNonTerminals; i++) {
        computeFirst(nonTerminals[i], i);
    }

    for (int i = 0; i < numNonTerminals; i++) {
        for (int j = 0; j < numTerminals; j++) {
            for (int k = 0; k < MAX_RULES; k++) {
                if (parseTable[i][terminals[j]][k] == 1) {
                    printf("M[%c, %c] = %s\n", nonTerminals[i], terminals[j], productions[i][k]);
                }
            }
        }
    }

    return 0;
}
