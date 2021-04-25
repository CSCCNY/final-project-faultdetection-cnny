import pdb
def createLevelArrayFromCircuitDescription(circuitDescription):
	
	inputs = circuitDescription[0]
	gates = circuitDescription[2]

	levels = [{
		i: {
			'leaving': None, 
			'entering': [k for k in gates if i in gates[k]['inputs']][0]
		} for i in inputs
	}]

	numberOfGatesDone = 0
	while numberOfGatesDone < len(gates):
		newLevel = {}
		# We check each gate, if all its inputs exist in some previous level, we add its output to the new level
		for gate in gates:
			if (all(any(i in level for level in levels) for i in gates[gate]['inputs'])):
				# all inputs belong to a previous level
				# we check that the outputs of the gate have not been added yet to the levels array
				if (all(o not in level for level in levels for o in gates[gate]['outputs'])):
					for o in gates[gate]['outputs']:
						entering = [k for k in gates if o in gates[k]['inputs']]
						newLevel[o] = {
							'leaving': gate,
							'entering': entering[0] if len(entering) > 0 else None
						}
					numberOfGatesDone += 1
		levels.append(newLevel)
	
	return levels