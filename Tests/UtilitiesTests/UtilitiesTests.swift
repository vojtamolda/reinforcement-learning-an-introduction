import XCTest
import OpenSpiel
import Utilities


final class UtilitiesTests: XCTestCase {
    func testAnyStochasticPolicy() {
        let ticTacToe = TicTacToe()

        let policies: [AnyStochasticPolicy<TicTacToe>] = [
            .init(UniformRandomPolicy<TicTacToe>(ticTacToe)),
            .init(UniformRandomPolicy<TicTacToe>(ticTacToe))
        ]
        
        for policy in policies {
            let t = type(of: policy)
            print(t)
        }
    }
}
