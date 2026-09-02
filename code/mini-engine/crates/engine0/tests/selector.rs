use engine0::model::Logits;
use engine0::{GenerationError, GreedySelector, Selector};

#[test]
fn greedy_selection_returns_the_largest_logit_index() {
    let logits = Logits::try_from_values(vec![-0.7, 0.1, 0.4, 2.2]).unwrap();
    assert_eq!(GreedySelector.select(&logits).unwrap().0, 3);
}

#[test]
fn greedy_selection_uses_first_index_as_the_tie_rule() {
    let logits = Logits::try_from_values(vec![-1.0, 2.0, 2.0, 0.0]).unwrap();
    assert_eq!(GreedySelector.select(&logits).unwrap().0, 1);
}

#[test]
fn empty_logits_fail_instead_of_inventing_a_token() {
    let logits = Logits::try_from_values(vec![]).unwrap();
    assert_eq!(
        GreedySelector.select(&logits),
        Err(GenerationError::NoLogits)
    );
}
