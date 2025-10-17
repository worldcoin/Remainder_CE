use remainder_frontend::{abstract_expr::AbstractExpression, layouter::nodes::NodeId};
use remainder_shared_types::Fr;

#[test]
fn test_abstract_expr_get_sources() {
    let node_id_1 = NodeId::new();
    let node_id_2 = NodeId::new();
    let node_id_3 = NodeId::new();

    let expression1 = AbstractExpression::constant(Fr::one());

    let expression2 = AbstractExpression::mle(node_id_1);

    let expression3 = expression1 - expression2;

    let expression4 = (expression3.clone()) * Fr::from(2);

    let expression5 = AbstractExpression::products(vec![node_id_2, node_id_3]);

    let expr = expression4.clone() + expression5.clone();

    assert_eq!(expr.get_sources(), vec![node_id_1, node_id_2, node_id_3]);

    assert_eq!(expression4.get_sources(), vec![node_id_1]);
    assert_eq!(expression4.get_sources(), expression3.get_sources());
    assert_eq!(expression5.get_sources(), vec![node_id_2, node_id_3]);
}
