/// A packed bit vector, one bit per row.
///
/// Both the validity (null) masks and the selection vectors produced by filters
/// are bitmaps. Packing them costs a shift and a mask per access but makes a
/// filter over a million rows touch 128 KB instead of 1 MB, and lets `and`
/// combine two predicates 64 rows at a time.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Bitmap {
    words: Vec<u64>,
    len: usize,
}

impl Bitmap {
    pub fn new() -> Self {
        Bitmap::default()
    }

    pub fn with_capacity(cap: usize) -> Self {
        Bitmap {
            words: Vec::with_capacity(cap.div_ceil(64)),
            len: 0,
        }
    }

    /// A bitmap of `len` bits all set to `value`.
    pub fn filled(len: usize, value: bool) -> Self {
        let mut words = vec![if value { u64::MAX } else { 0 }; len.div_ceil(64)];
        if value {
            trim_tail(&mut words, len);
        }
        Bitmap { words, len }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn push(&mut self, value: bool) {
        if self.len % 64 == 0 {
            self.words.push(0);
        }
        if value {
            let (w, b) = (self.len / 64, self.len % 64);
            self.words[w] |= 1u64 << b;
        }
        self.len += 1;
    }

    /// Reads bit `i`. Out-of-range reads return false rather than panicking:
    /// callers reach for this in tight loops where a bounds check per row is
    /// exactly the cost the packed representation is meant to avoid.
    pub fn get(&self, i: usize) -> bool {
        if i >= self.len {
            return false;
        }
        self.words[i / 64] & (1u64 << (i % 64)) != 0
    }

    pub fn set(&mut self, i: usize, value: bool) {
        debug_assert!(i < self.len);
        let (w, b) = (i / 64, i % 64);
        if value {
            self.words[w] |= 1u64 << b;
        } else {
            self.words[w] &= !(1u64 << b);
        }
    }

    /// Number of set bits — the selectivity of a filter, or the non-null count
    /// of a column. `count_ones` is a single instruction per 64 rows.
    pub fn count_set(&self) -> usize {
        self.words.iter().map(|w| w.count_ones() as usize).sum()
    }

    pub fn count_unset(&self) -> usize {
        self.len - self.count_set()
    }

    pub fn all_set(&self) -> bool {
        self.count_set() == self.len
    }

    pub fn any_set(&self) -> bool {
        self.words.iter().any(|w| *w != 0)
    }

    /// Word-at-a-time intersection, used to combine conjunctive predicates.
    pub fn and(&self, other: &Bitmap) -> Bitmap {
        let len = self.len.min(other.len);
        let mut words: Vec<u64> = self
            .words
            .iter()
            .zip(other.words.iter())
            .map(|(a, b)| a & b)
            .collect();
        words.truncate(len.div_ceil(64));
        trim_tail(&mut words, len);
        Bitmap { words, len }
    }

    pub fn or(&self, other: &Bitmap) -> Bitmap {
        let len = self.len.max(other.len);
        let mut words = vec![0u64; len.div_ceil(64)];
        for (i, w) in words.iter_mut().enumerate() {
            let a = self.words.get(i).copied().unwrap_or(0);
            let b = other.words.get(i).copied().unwrap_or(0);
            *w = a | b;
        }
        trim_tail(&mut words, len);
        Bitmap { words, len }
    }

    pub fn not(&self) -> Bitmap {
        let mut words: Vec<u64> = self.words.iter().map(|w| !w).collect();
        trim_tail(&mut words, self.len);
        Bitmap {
            words,
            len: self.len,
        }
    }

    /// The positions of the set bits, which is what the take-based operators
    /// (join, sort) consume.
    pub fn set_indices(&self) -> Vec<usize> {
        let mut out = Vec::with_capacity(self.count_set());
        for (wi, mut word) in self.words.iter().copied().enumerate() {
            while word != 0 {
                let bit = word.trailing_zeros() as usize;
                let idx = wi * 64 + bit;
                if idx < self.len {
                    out.push(idx);
                }
                word &= word - 1;
            }
        }
        out
    }

    pub fn iter(&self) -> impl Iterator<Item = bool> + '_ {
        (0..self.len).map(move |i| self.get(i))
    }
}

impl FromIterator<bool> for Bitmap {
    fn from_iter<T: IntoIterator<Item = bool>>(iter: T) -> Self {
        let mut b = Bitmap::new();
        for v in iter {
            b.push(v);
        }
        b
    }
}

/// Clears the bits past `len` in the final word so `count_ones` cannot count
/// padding. Every operation that fills whole words has to call this.
fn trim_tail(words: &mut [u64], len: usize) {
    let rem = len % 64;
    if rem != 0 {
        if let Some(last) = words.last_mut() {
            *last &= (1u64 << rem) - 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_and_get_round_trip_across_a_word_boundary() {
        let pattern: Vec<bool> = (0..130).map(|i| i % 3 == 0).collect();
        let bm: Bitmap = pattern.iter().copied().collect();
        assert_eq!(bm.len(), 130);
        for (i, expected) in pattern.iter().enumerate() {
            assert_eq!(bm.get(i), *expected, "bit {i}");
        }
    }

    #[test]
    fn filled_does_not_count_padding_bits() {
        // 70 bits spans two words; the upper 58 bits of the second must not
        // show up in count_set.
        let bm = Bitmap::filled(70, true);
        assert_eq!(bm.count_set(), 70);
        assert!(bm.all_set());
        assert_eq!(Bitmap::filled(70, false).count_set(), 0);
    }

    #[test]
    fn and_or_not_agree_with_bit_by_bit_evaluation() {
        let a: Bitmap = (0..100).map(|i| i % 2 == 0).collect();
        let b: Bitmap = (0..100).map(|i| i % 3 == 0).collect();
        let and = a.and(&b);
        let or = a.or(&b);
        let not = a.not();
        for i in 0..100 {
            assert_eq!(and.get(i), a.get(i) && b.get(i));
            assert_eq!(or.get(i), a.get(i) || b.get(i));
            assert_eq!(not.get(i), !a.get(i));
        }
        assert_eq!(not.len(), 100);
    }

    #[test]
    fn negation_does_not_leak_padding_into_the_count() {
        let bm = Bitmap::filled(5, false);
        assert_eq!(bm.not().count_set(), 5);
    }

    #[test]
    fn set_indices_lists_exactly_the_set_positions() {
        let mut bm = Bitmap::filled(200, false);
        for i in [0, 63, 64, 65, 199] {
            bm.set(i, true);
        }
        assert_eq!(bm.set_indices(), vec![0, 63, 64, 65, 199]);
        assert_eq!(bm.count_set(), 5);
        assert_eq!(bm.count_unset(), 195);
    }

    #[test]
    fn set_can_clear_a_bit_again() {
        let mut bm = Bitmap::filled(10, true);
        bm.set(4, false);
        assert!(!bm.get(4));
        assert_eq!(bm.count_set(), 9);
    }

    #[test]
    fn reads_past_the_end_are_false_rather_than_a_panic() {
        let bm = Bitmap::filled(3, true);
        assert!(!bm.get(3));
        assert!(!bm.get(9999));
    }

    #[test]
    fn empty_bitmap_behaves() {
        let bm = Bitmap::new();
        assert!(bm.is_empty());
        assert!(!bm.any_set());
        assert!(bm.all_set()); // vacuously
        assert!(bm.set_indices().is_empty());
        assert_eq!(Bitmap::with_capacity(128).len(), 0);
    }

    #[test]
    fn and_truncates_to_the_shorter_input() {
        let a = Bitmap::filled(100, true);
        let b = Bitmap::filled(30, true);
        let r = a.and(&b);
        assert_eq!(r.len(), 30);
        assert_eq!(r.count_set(), 30);
    }

    #[test]
    fn or_extends_to_the_longer_input() {
        let a = Bitmap::filled(10, true);
        let b = Bitmap::filled(100, false);
        let r = a.or(&b);
        assert_eq!(r.len(), 100);
        assert_eq!(r.count_set(), 10);
        assert!(r.any_set());
    }

    #[test]
    fn iter_yields_every_bit_in_order() {
        let bits = vec![true, false, true];
        let bm: Bitmap = bits.iter().copied().collect();
        assert_eq!(bm.iter().collect::<Vec<_>>(), bits);
    }
}
