// gray-codes-rs: src/lib.rs
//
// Copyright 2017 David Creswick
//
// Licensed under the Apache License, Version 2.0, <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
// implied.  See the License for the specific language governing
// permissions and limitations under the License.

//! Gray code iterators and related utilities.
//!
//! A Gray code is a reordering of the integers such that adjacent
//! codes differ by exactly one bit.
//!
//! The `GrayCode{8,16,32,64}` structs provide iterators over binary
//! reflected Gray codes in various unsigned integer sizes as well as
//! direct conversions to and from the codes. The `Subsets` struct
//! provides a convenient way to iterate over subsets of a slice. The
//! `InclusionExclusion` struct provides a building block for visiting
//! all subsets more efficiently.
use std::iter::{self, FromIterator};
use std::option;
use std::ops::Range;


// FIXME: write tests that cover all the "Panics" sections


macro_rules! gray_code_impl {
    (#[$doc:meta] $nm:ident, $uint_ty:ty, $bits:expr, $test_name:ident, $test_bits:expr) => {
        #[$doc]
        ///
        ///# Examples
        ///
        /// Generate all four-bit Gray codes.
        ///
        /// ```
        /// use gray_codes::GrayCode32;
        ///
        /// assert_eq!(GrayCode32::with_bits(4).collect::<Vec<u32>>(),
        ///            vec![0b0000,
        ///                 0b0001,
        ///                 0b0011,
        ///                 0b0010,
        ///                 0b0110,
        ///                 0b0111,
        ///                 0b0101,
        ///                 0b0100,
        ///                 0b1100,
        ///                 0b1101,
        ///                 0b1111,
        ///                 0b1110,
        ///                 0b1010,
        ///                 0b1011,
        ///                 0b1001,
        ///                 0b1000]);
        /// ```
        ///
        /// This could also be done with either of the other two constructors.
        ///
        /// ```
        /// # use gray_codes::GrayCode32;
        /// #
        /// for (x,y) in Iterator::zip(GrayCode32::with_bits(4),
        ///                            GrayCode32::over_range(0..(1<<4))) {
        ///     assert!(x == y);
        /// }
        ///
        /// for (x,y) in Iterator::zip(GrayCode32::with_bits(4),
        ///                            GrayCode32::all().take(1<<4)) {
        ///     assert!(x == y);
        /// }
        /// ```
        #[derive(Clone, Debug)] // FIXME: it might be nice if this were PartialEq+Eq
        pub struct $nm {
            range: iter::Chain<Range<$uint_ty>, option::IntoIter<$uint_ty>>
        }

        impl $nm {
            /// Construct an iterator over n-bit Gray codes.
            ///
            /// # Panics
            ///
            /// Panics if `bits` is larger than the unsigned integer type.
            pub fn with_bits(bits: usize) -> $nm {
                assert!(bits <= $bits);
                $nm {
                    range: if bits == $bits {
                        (0..<$uint_ty>::max_value()).chain(Some(<$uint_ty>::max_value()).into_iter())
                    } else {
                        (0..((1 as $uint_ty) << bits)).chain(None.into_iter())
                    }
                }
            }

            /// Construct an iterator over a specific range of Gray codes.
            ///
            /// The range bounds need not be powers of 2.
            pub fn over_range(range: Range<$uint_ty>) -> $nm {
                $nm {
                    range: range.chain(None.into_iter())
                }
            }

            /// Construct an iterator over all Gray codes.
            ///
            /// This iterator yields all codes that fit in the
            /// unsigned integer type.
            #[inline]
            pub fn all() -> $nm {
                $nm::with_bits($bits)
            }

            /// Convert a binary value to the corresponding Gray code.
            #[inline]
            pub fn from_index(val: $uint_ty) -> $uint_ty {
                val ^ (val >> 1)
            }

            /// Convert a Gray code from the corresponding binary value.
            #[inline]
            pub fn to_index(code: $uint_ty) -> $uint_ty {
                let mut val = code;
                // After macro expansion, these conditionals are
                // constant and so will be optimized out, depending on
                // the size of the unsigned  integer type.
                if $bits > 32 {
                    val = val ^ val.wrapping_shr(32);
                }
                if $bits > 16 {
                    val = val ^ val.wrapping_shr(16);
                }
                if $bits > 8 {
                    val = val ^ val.wrapping_shr(8);
                }
                val = val ^ (val >> 4);
                val = val ^ (val >> 2);
                val = val ^ (val >> 1);
                val
            }

            /// Compute the next Gray code by flipping a single bit of
            /// the argument.
            pub fn next_code(code: $uint_ty) -> Option<$uint_ty> {
                let j;
                if code.count_ones() & 1 == 0 {
                    j = 0;
                } else {
                    j = code.trailing_zeros() + 1;
                    if j == $bits {
                        return None;
                    }
                }
                return Some(code ^ (1<<j));
            }
        }

        impl Iterator for $nm {
            type Item = $uint_ty;

            fn next(&mut self) -> Option<$uint_ty> {
                self.range.next().map($nm::from_index)
            }

            fn size_hint(&self) -> (usize, Option<usize>) {
                self.range.size_hint()
            }
        }

        impl DoubleEndedIterator for $nm {
            fn next_back(&mut self) -> Option<$uint_ty> {
                self.range.next_back().map($nm::from_index)
            }
        }

        #[test]
        fn $test_name() {
            // Each Gray code differs from the previous code by one bit.
            {
                let mut it = $nm::with_bits($test_bits);
                let mut prev = it.next().unwrap();
                for this in it {
                    assert!((this ^ prev).count_ones() == 1);
                    prev = this;
                }
            }
            // And every code in 0..(1<<N) is returned exactly once.
            let codes: Vec<$uint_ty> = $nm::with_bits($test_bits).collect();
            assert!(codes.len() == (1<<$test_bits));
            assert!((0..(1<<$test_bits)).all(|n| codes.contains(&(n as $uint_ty))));

            // test the .from_index() and .to_index() methods
            for (idx, code) in $nm::with_bits($test_bits).enumerate() {
                let idx = idx as $uint_ty;
                assert_eq!($nm::from_index(idx), code);
                assert_eq!($nm::to_index(code), idx);
            }

            // test the .next_code() method
            for (code, next_code) in Iterator::zip($nm::with_bits($test_bits),
                                                   $nm::all().skip(1)) {
                assert!($nm::next_code(code) == Some(next_code));
            }
        }
    }
}

gray_code_impl!(#[doc="Iterator over binary reflected Gray codes as u8 values"]
                GrayCode8, u8, 8, gray_code_8, 8);
gray_code_impl!(#[doc="Iterator over binary reflected Gray codes as u16 values"]
                GrayCode16, u16, 16, gray_code_16, 10);
gray_code_impl!(#[doc="Iterator over binary reflected Gray codes as u32 values"]
                GrayCode32, u32, 32, gray_code_32, 11);
gray_code_impl!(#[doc="Iterator over binary reflected Gray codes as u64 values"]
                GrayCode64, u64, 64, gray_code_64, 12);


/// Set mutation operations.
///
/// See the `InclusionExclusion` struct.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SetMutation {
    /// Add the item with the given index to the set.
    Insert(usize),
    /// Remove the item with the given index from the set.
    Remove(usize)
}


/// Iterator yielding `SetMutation`s, allowing the efficient
/// construction of all subsets of n items
///
/// The iterator yields `SetMutation` operations, which instruct one
/// to either include or exclude a single item by index. Starting from
/// an empty set, the instructions will cause every subset of n items
/// to be visited exactly once.
///
/// # Examples
///
/// Visit every subset of a set of strings by mutating a `HashSet`
/// exactly once per iteration.
///
/// ```
/// use std::collections::HashSet;
/// use gray_codes::{InclusionExclusion, SetMutation};
///
/// static STRINGS: &[&str] = &["apple", "moon", "pie"];
///
/// let mut subset = HashSet::with_capacity(STRINGS.len());
/// // Visit the empty set here, if desired.
/// println!("{:?}", subset);
/// for mutation in InclusionExclusion::of_len(STRINGS.len()) {
///     // Mutate the set, according to instructions.
///     match mutation {
///         SetMutation::Insert(i) => { subset.insert(STRINGS[i]); },
///         SetMutation::Remove(i) => { subset.remove(STRINGS[i]); },
///     }
///     // Visit the never-before-seen subset here.
///     println!("{:?}", subset);
/// }
/// ```
///
/// Iterate over the 15 possible ways to sum four numbers using only a
/// single addition or subtraction per iteration.
///
/// ```
/// use gray_codes::{InclusionExclusion, SetMutation};
///
/// let addends = [235, 63, 856, 967];
///
/// let mut sum = 0;
/// for mutation in InclusionExclusion::of_len(addends.len()) {
///     match mutation {
///         SetMutation::Insert(i) => { sum += addends[i]; },
///         SetMutation::Remove(i) => { sum -= addends[i]; }
///     }
///     // Process the sum somehow.
///     println!("{}", sum);
/// }
/// ```
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct InclusionExclusion {
    focus_ptrs: Vec<u16>,
    current_set: Vec<bool>
}

impl InclusionExclusion {
    /// Constructor for an `InclusionExclusion` iterator over the
    /// given number of objects.
    ///
    /// See the struct documentation for examples.
    ///
    /// # Panics
    ///
    /// This method panics if either of the following conditions are
    /// true:
    ///
    ///  - `item_count` is zero.
    ///  - `item_count` uses more than 16 bits.
    ///
    pub fn of_len(item_count: usize) -> InclusionExclusion {
        assert!(item_count > 0);
        assert!(item_count <= u16::max_value() as usize);
        InclusionExclusion {
            focus_ptrs: (0..item_count as u16).collect(),
            current_set: vec![false; item_count]
        }
    }
}

impl Iterator for InclusionExclusion {
    type Item = SetMutation;

    fn next(&mut self) -> Option<SetMutation> {
        let j = self.focus_ptrs[0] as usize;
        if j == self.focus_ptrs.len() {
            None
        } else {
            self.focus_ptrs[0] = 0;
            if j+1 == self.focus_ptrs.len() {
                self.focus_ptrs[j] = self.focus_ptrs.len() as u16;
            } else {
                self.focus_ptrs[j] = self.focus_ptrs[j+1];
                self.focus_ptrs[j+1] = j as u16 + 1;
            }
            // Flip an internal bit and return the corresponding
            // instruction.
            self.current_set[j] ^= true;
            if self.current_set[j] {
                Some(SetMutation::Insert(j))
            } else {
                Some(SetMutation::Remove(j))
            }
        }
    }
}

#[test]
fn inclusion_exclusion() {
    const N: usize = 12;
    assert!(InclusionExclusion::of_len(N).count() == (1<<N)-1);
    // Use the bits of a u32 to represent set inclusion/exclusion.
    // This should match integer-valued gray codes exactly.
    let mut x: u32 = 0;
    let mut gray_codes = GrayCode32::with_bits(N);
    assert!(gray_codes.next() == Some(0));
    for (op, code) in Iterator::zip(InclusionExclusion::of_len(N),
                                    gray_codes) {
        match op {
            SetMutation::Insert(i) => {
                // Make sure it *isn't* already included first.
                assert!(x & (1<<i) == 0);
                x |= 1<<i;
            },
            SetMutation::Remove(i) => {
                // Make sure it *is* already included first.
                assert!(x & (1<<i) != 0);
                x &= !(1<<i);
            }
        }
        assert!(x == code);
    }
}


/// Iterator yielding subsets of a slice
///
/// Use the static method `Subsets::of(...)` to construct the iterator.
///
/// The input is a slice of type `&'a [T]` and the output is any
/// container `C` that is `FromIterator<&'a T>`. In many cases, it's
/// good enough for the collection `C` to be `Vec<&'a T>`, in which
/// case you can use the convenient `VecSubsets` type alias.
///
/// A new `C`-container is created every iteration, which is an
/// O(set_len) operation per iteration. Greater efficiency (O(1) per
/// iteration) can be gained by using the `InclusionExclusion`
/// iterator to perform mutation directly on your own container.
///
/// # Examples
///
/// Collect every subset of `0..4` into a `Vec` of `Vec`s.
///
/// ```
/// use gray_codes::VecSubsets;
/// static NUMBERS: &[u32] = &[0,1,2,3];
/// let mut subsets: Vec<_> = VecSubsets::of(NUMBERS).collect();
/// assert!(subsets.len() == 16);
/// subsets.sort();
/// // (Note that this is sorted order, not the order in which the
/// // Subsets iterator generates items.)
/// assert!(subsets == vec![vec![],
///                         vec![&0],
///                         vec![&0,&1],
///                         vec![&0,&1,&2],
///                         vec![&0,&1,&2,&3],
///                         vec![&0,&1,&3],
///                         vec![&0,&2],
///                         vec![&0,&2,&3],
///                         vec![&0,&3],
///                         vec![&1],
///                         vec![&1,&2],
///                         vec![&1,&2,&3],
///                         vec![&1,&3],
///                         vec![&2],
///                         vec![&2,&3],
///                         vec![&3]]);
/// ```
///
/// Collect every subset of characters from the word "cat" into a
/// `HashSet` of `String`s.
///
/// ```
/// # use gray_codes::Subsets;
/// # use std::collections::HashSet;
/// static CHARS: &[char] = &['c', 'a', 't'];
/// let subsets: HashSet<String> = Subsets::of(CHARS).collect();
/// assert!(subsets.len() == 8);
/// // (Note that this is not the order in which the Subsets iterator
/// // generates items. It is merely convenient for checking the results.)
/// assert!(subsets.contains(""));
/// assert!(subsets.contains("c"));
/// assert!(subsets.contains("a"));
/// assert!(subsets.contains("t"));
/// assert!(subsets.contains("ca"));
/// assert!(subsets.contains("at"));
/// assert!(subsets.contains("ct"));
/// assert!(subsets.contains("cat"));
/// ```
#[derive(Clone, Debug)]
pub struct Subsets<'a, T:'a, C> {
    items: &'a [T],
    next: Option<C>,
    inc_ex: InclusionExclusion
}

impl<'a, T:'a, C: FromIterator<&'a T>> Subsets<'a, T, C> {
    /// Constructor.
    pub fn of(items: &'a [T]) -> Subsets<'a, T, C> {
        Subsets {
            items: items,
            next: Some(iter::empty().collect()),
            inc_ex: InclusionExclusion::of_len(items.len())
        }
    }
}

impl<'a, T:'a, C: FromIterator<&'a T>> Iterator for Subsets<'a, T, C> {
    type Item = C;

    fn next(&mut self) -> Option<C> {
        let ret = self.next.take();
        if let Some(_) = self.inc_ex.next() {
            let collection = self.inc_ex.current_set.iter().enumerate().flat_map(|(i,&b)| {
                if b {
                    Some(&self.items[i])
                } else {
                    None
                }
            }).collect();
            self.next = Some(collection);
        }
        return ret;
    }
}

/// Alias for iterating over `Vec`-valued subsets of a slice.
pub type VecSubsets<'a,T> = Subsets<'a, T, Vec<&'a T>>;

#[test]
fn subsets() {
    static ITEMS: &[char] =  &['a', 'b', 'c', 'd', 'e'];
    let mut subsets_seen = Vec::new();
    for subset in VecSubsets::of(ITEMS) {
        assert!(!subsets_seen.contains(&subset));
        subsets_seen.push(subset);
    }
    assert!(subsets_seen.len() == 1<<ITEMS.len());
}
