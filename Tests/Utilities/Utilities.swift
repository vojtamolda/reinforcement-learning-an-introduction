import TensorFlow


public extension Array where Element: Comparable {
    func argMax() -> Array.Index? {
        let maxEnumerated = enumerated().max { $0.element > $1.element }
        return maxEnumerated?.offset
    }
}

public extension Dictionary where Value == Double {
    /// Sample an action from a dictionary that represents probabilities of a discrete action distribution.
    func sample() -> Key? {
        precondition(!self.isEmpty)
        
        let probabilitiesSum = values.reduce(0.0, +)
        let random = Tensor(randomUniform: [1], lowerBound: Tensor(0.0),
                            upperBound: Tensor(probabilitiesSum)).scalarized()
        
        var sampledAction = nil as Key?
        var accumulatedProbability = 0.0
        for (action, probability) in self {
            sampledAction = action
            accumulatedProbability += probability
            if random < accumulatedProbability { break }
        }
        
        return sampledAction
    }
}
