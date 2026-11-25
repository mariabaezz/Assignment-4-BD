"""Independent O(n^2) checksum of the deterministic matrix product."""
import argparse


def sums(n, seed, columns):
    result = [0] * n
    x = seed
    for i in range(n):
        for j in range(n):
            x = (1664525 * x + 1013904223) & 0xffffffff
            result[j if columns else i] += x % 10
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('n', type=int, nargs='?', default=4000)
    args = parser.parse_args()
    if args.n <= 0:
        parser.error('n must be positive')
    a = sums(args.n, 42, True)
    b = sums(args.n, 4242, False)
    print(f'n={args.n},checksum={sum(x*y for x,y in zip(a,b))}')


if __name__ == '__main__':
    main()
