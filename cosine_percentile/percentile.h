#ifndef PERCENTILE_H
#define PERCENTILE_H

/* percentile.h -- percentile-based outlier clipping for weight matrices.
 *
 * Used after weight quantisation (or before, to precondition W) to
 * suppress extreme weight magnitudes that would otherwise dominate the
 * quantisation grid. For each output channel o we compute the 99.95th
 * percentile of |W[o, :]| and clamp every entry of that channel to
 * [-clip_val, +clip_val].
 *
 * Pure C11, no external libraries. Compile with:
 *   gcc -O3 -march=native -fopenmp -std=c11 -c percentile.c -o percentile.o
 */

/* Percentile of a *sorted* (ascending) array.
 *
 *   sorted_arr : float array of length n, sorted ascending. The function
 *                does NOT sort; the caller is responsible for sorting
 *                (e.g. via qsort) so that percentile() can run in O(1).
 *   n          : number of elements
 *   pct        : percentile as a FRACTION in [0, 1]. 0 -> min, 0.5 ->
 *                median, 1 -> max. Values outside [0, 1] are clamped.
 *
 * Indexing convention: linear interpolation between the two nearest
 * samples, identical to numpy.quantile with interpolation='linear'.
 *
 *   rank  = pct * (n - 1)
 *   lo    = floor(rank)
 *   hi    = ceil(rank)
 *   frac  = rank - lo
 *   value = sorted_arr[lo] + frac * (sorted_arr[hi] - sorted_arr[lo])
 *
 * For n == 1, returns sorted_arr[0] for any pct. For n <= 0 returns 0.0.
 */
float percentile(float *sorted_arr, int n, float pct);

/* Clip outliers in each output channel of W (out_dim x in_dim, row-major).
 *
 * For each output channel o:
 *   1. Copy |W[o, :]| into a temporary buffer of length in_dim.
 *   2. qsort it ascending.
 *   3. clip_val = percentile(temp, in_dim, PERCENTILE_CLIP_PCT).
 *   4. For every i: if |W[o, i]| > clip_val, replace W[o, i] with
 *      copysign(clip_val, W[o, i]).
 *
 * Returns: the number of output channels where at least one element was
 *          actually clamped (i.e. |W[o, i]| > clip_val for some i).
 *
 * The clip threshold defaults to the 99.95th percentile (pct = 0.9995)
 * and may be overridden at compile time:
 *
 *   gcc -DPERCENTILE_CLIP_PCT=0.999 ...
 *
 * Channels whose |W[o,:]| are all within the clip threshold are left
 * untouched; their values are not even read a second time on the clamp
 * pass (an early-exit test per channel avoids the second sweep when no
 * element exceeds the threshold).
 */
int clip_outliers(float *W, int out_dim, int in_dim);

#ifndef PERCENTILE_CLIP_PCT
#define PERCENTILE_CLIP_PCT 0.9995f
#endif

#endif /* PERCENTILE_H */
