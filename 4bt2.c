#include <stdio.h>

int main() {
  int width, height;
  int pixel;
  int hist[256] = {0};

  printf("Enter width and height: ");
  scanf("%d %d", &width, &height);

  printf("Enter pixel values (0-255):\n");

  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      scanf("%d", &pixel);

      if (pixel >= 0 && pixel <= 255) {
        hist[pixel]++;
      }
    }
  }

  printf("\nHistogram:\n");
  for (int i = 0; i < 256; i++) {
    if (hist[i] > 0) {
      printf("Gray level %d: %d\n", i, hist[i]);
    }
  }

  return 0;
}