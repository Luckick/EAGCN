# Generate Graphs and Atom Features Matrix (afm), Adjacent Matrix (adj), and relation tensors for each molecule.

import numpy as np
import rdkit.Chem.AllChem as AllChem
import rdkit.Chem.Descriptors as Descriptors
import rdkit.Chem.rdMolDescriptors as rdMolDescriptors
import rdkit.Chem.EState as EState
import rdkit.Chem.rdPartialCharges as rdPartialCharges
import rdkit.Chem.rdChemReactions as rdRxns
import copy
from rdkit import Chem
att_dtype = np.float32

echo = True

class Graph():
	'''Describes an undirected graph class'''
	def __init__(self):
		self.nodes = []
		self.num_nodes = 0
		self.edges = []
		self.num_edges = 0
		self.N_features = 0
		self.bondtype_list_order = []
		self.atomtype_list_order = []
		return

	def nodeAttributes(self):
		'''Returns 2D array where (#, :) contains attributes of node #'''
		return (np.vstack([x.attributes for x in self.nodes]))

	def edgeAttributes(self):
		'''Returns 2D array where (#, :) contains attributes of edge #'''
		return (np.vstack([x.attributes for x in self.edges]))

	def edgeTypeAtt(self):
		return (np.vstack([x.TypeAtt for x in self.edges]))

	def nodeNeighbors(self):
		return [x.neighbors for x in self.nodes]

	def clone(self):
		'''clone() method to trick Theano'''
		return copy.deepcopy(self)
	"""
	# Add the AdjTensor with edge info
	def getAdjTensor(self, maxNodes):
		adjTensor = np.zeros([maxNodes, maxNodes, self.edgeFeatureDim + 1])
		for edge in self.edges:
			(i, j) = edge.ends
			adjTensor[i, j, 0] = 1.0
			adjTensor[j, i, 0] = 1.0
			adjTensor[i, j, 1:] = edge.features
			adjTensor[j, i, 1:] = edge.features
		return adjTensor
	"""
	def dump_as_matrices_Att(self):
		# Bad input handling
		if not self.nodes:
			raise GraphError('Error generating tensor for graph with no nodes')
		if not self.edges:
			raise GraphError('Need at least one bond!')

		N_nodes = len(self.nodes)

		F_a, F_bAtt = sizeAttributeVectorsAtt(self.bondtype_list_order, self.atomtype_list_order,
											  molecular_attributes=self.molecular_attributes)

		mat_features = np.zeros((N_nodes, F_a), dtype = np.float32)

		mat_adjacency = np.zeros((N_nodes, N_nodes), dtype = np.float32)
		adjTensor_TypeAtt = np.zeros([F_bAtt, N_nodes, N_nodes], dtype = np.float32)

		adjTensor_OrderAtt = np.zeros([4, N_nodes, N_nodes], dtype = np.float32)
		adjTensor_AromAtt = np.zeros([2, N_nodes, N_nodes], dtype = np.float32)
		adjTensor_ConjAtt = np.zeros([2, N_nodes, N_nodes], dtype = np.float32)
		adjTensor_RingAtt = np.zeros([2, N_nodes, N_nodes], dtype=np.float32)

		edgeAttributes = np.vstack([x.attributes for x in self.edges])
		nodeAttributes = np.vstack([x.attributes for x in self.nodes])
		nodeSubtypes = np.vstack([x.subtype for x in self.nodes])

		for i, node in enumerate(self.nodes):
			mat_features[i, :] = nodeAttributes[i]
			mat_adjacency[i, i] = 0 # include self terms
			# Delete this line to elininate self attention. will set 1 to all self.
			#adjTensorAtt[0:len(self.atomtype_list_order), i, i] = node.TypeAtt # diagonal element. self attention.
			#  # set to zero, so we can add identity latter (in the training process after dictionary mapped to att_matrices.)
			adjTensor_TypeAtt[0, i, i] = 0
			adjTensor_OrderAtt[0, i, i] = 0
			adjTensor_AromAtt[0,i,i] = 0
			adjTensor_ConjAtt[0, i, i] = 0
			adjTensor_RingAtt[0, i, i] = 0
		"""
		for e, edge in enumerate(self.edges):
			(i, j) = edge.connects
			mat_adjacency[i, j] = 1.0
			mat_adjacency[j, i] = 1.0

			# Keep track of extra special bond types - which are nothing more than
			# bias terms specific to the bond type because they are all one-hot encoded
			#mat_specialbondtypes[i, :] += edgeAttributes[e]
			#mat_specialbondtypes[j, :] += edgeAttributes[e]
		"""
		for edge in self.edges:
			(i, j) = edge.connects
			#adjTensorAtt[len(self.atomtype_list_order):, i, j] = edge.TypeAtt # non-diagonal element, neighbor attention.
			#adjTensorAtt[len(self.atomtype_list_order):, j, i] = edge.TypeAtt
			mat_adjacency[i, j] = 1
			mat_adjacency[j, i] = 1
			adjTensor_TypeAtt[0:, i, j] = edge.TypeAtt
			adjTensor_TypeAtt[0:, j, i] = edge.TypeAtt
			adjTensor_OrderAtt[0:, i, j] = edge.orderAtt
			adjTensor_OrderAtt[0:, j, i] = edge.orderAtt
			adjTensor_AromAtt[0:, i, j] = edge.aromAtt
			adjTensor_AromAtt[0:, j, i] = edge.aromAtt
			adjTensor_ConjAtt[0:, i, j] = edge.conjAtt
			adjTensor_ConjAtt[0:, j, i] = edge.conjAtt
			adjTensor_RingAtt[0:, i, j] = edge.ringAtt
			adjTensor_RingAtt[0:, j, i] = edge.ringAtt

		return (mat_features, mat_adjacency, adjTensor_TypeAtt, adjTensor_OrderAtt, adjTensor_AromAtt, adjTensor_ConjAtt, adjTensor_RingAtt, nodeSubtypes) # replace mat_specialbondtypes bt adjTensor

class Node():
	'''Describes an attributed node in an undirected graph'''
	def __init__(self, i = None, attributes = np.array([], dtype = att_dtype), subtype = np.array([], dtype = att_dtype),
				 TypeAtt = np.array([], dtype = att_dtype)):
		self.i = i
		self.attributes = attributes # 1D array
		self.neighbors = [] # (atom index, bond index)
		self.TypeAtt = TypeAtt
		self.subtype = subtype
		return

class Edge():
	'''Describes an attributed edge in an undirected graph'''
	def __init__(self, connects = (), i = None, attributes = np.array([], dtype = att_dtype),
				 TypeAtt = np.array([], dtype = att_dtype), orderAtt = np.array([], dtype = att_dtype),
				 aromAtt=np.array([], dtype=att_dtype), conjAtt = np.array([], dtype = att_dtype),
				 ringAtt = np.array([], dtype = att_dtype)):
		self.i = i
		self.attributes = attributes # 1D array
		self.TypeAtt = TypeAtt
		self.connects = connects # (atom index, atom index)
		self.orderAtt = orderAtt
		self.aromAtt = aromAtt
		self.conjAtt = conjAtt
		self.ringAtt = ringAtt
		return


def molToGraph(rdmol, bondtype_list_order, atomtype_list_order, selected_atom_list=[5, 6, 7, 8, 9, 15, 16, 17, 35, 53],
			   molecular_attributes = False):
	'''Converts an RDKit molecule to an attributed undirected graph'''
	# Initialize
	graph = Graph()
	graph.molecular_attributes = molecular_attributes
	graph.bondtype_list_order = bondtype_list_order
	bond_list = bondtype_list_order
	graph.atomtype_list_order = atomtype_list_order
	rdmol = sub_type_setup(rdmol)
	subtype_list = ['os', 'oh', 'oa', 'o', 'un O', 'c3', 'c1', 'ca', 'c', 'c2', 'un C', 'n1', 'n3', 'na', 'n', 'no',
					'nh', 'un N', 's2', '-sh', '-ss', 'un S', 'f', 'cl', 'br', 'i', 'p', 'unreconginized']
	# Calculate atom-level molecule descriptors
	attributes = [[] for i in rdmol.GetAtoms()]
	"""
	if molecular_attributes:
		labels = []
		[attributes[i].append(x[0]) \
			for (i, x) in enumerate(rdMolDescriptors._CalcCrippenContribs(rdmol))]
		labels.append('Crippen contribution to logp')

		[attributes[i].append(x[1]) \
			for (i, x) in enumerate(rdMolDescriptors._CalcCrippenContribs(rdmol))]
		labels.append('Crippen contribution to mr')

		[attributes[i].append(x) \
			for (i, x) in enumerate(rdMolDescriptors._CalcTPSAContribs(rdmol))]
		labels.append('TPSA contribution')

		[attributes[i].append(x) \
			for (i, x) in enumerate(rdMolDescriptors._CalcLabuteASAContribs(rdmol)[0])]
		labels.append('Labute ASA contribution')

		[attributes[i].append(x) \
			for (i, x) in enumerate(EState.EStateIndices(rdmol))]
		labels.append('EState Index')

		rdPartialCharges.ComputeGasteigerCharges(rdmol)
		[attributes[i].append(float(a.GetProp('_GasteigerCharge'))) \
			for (i, a) in enumerate(rdmol.GetAtoms())]
		labels.append('Gasteiger partial charge')

		# Gasteiger partial charges sometimes gives NaN
		for i in range(len(attributes)):
			if np.isnan(attributes[i][-1]) or np.isinf(attributes[i][-1]):
				attributes[i][-1] = 0.0

		[attributes[i].append(float(a.GetProp('_GasteigerHCharge'))) \
			for (i, a) in enumerate(rdmol.GetAtoms())]
		labels.append('Gasteiger hydrogen partial charge')

		# Gasteiger partial charges sometimes gives NaN
		for i in range(len(attributes)):
			if np.isnan(attributes[i][-1]) or np.isinf(attributes[i][-1]):
				attributes[i][-1] = 0.0
	"""
	# Add bonds: only for real bonded atoms, self not included.
	for bond in rdmol.GetBonds():
		edge = Edge()
		edge.i = bond.GetIdx()
		# this edge attributes will be replaced by 5 relations.
		#edge.attributes = bondAttributes(bond)
		edge.orderAtt = list(oneHotVector(bond.GetBondTypeAsDouble(), [1.0, 1.5, 2.0, 3.0]))
		edge.aromAtt = list(oneHotVector(bond.GetIsAromatic(), [1.0, 0.0]))
		edge.conjAtt = list(oneHotVector(bond.GetIsConjugated(), [1.0, 0.0]))
		edge.ringAtt = list(oneHotVector(bond.IsInRing(), [1.0, 0.0]))

		BeginAtom, EndAtom = bond.GetBeginAtom(), bond.GetEndAtom()
		begin_idx, end_idx = BeginAtom.GetAtomicNum(), EndAtom.GetAtomicNum()
		if begin_idx < end_idx:
			bond_type = str(begin_idx) + '_' + str(end_idx)
		else:
			bond_type= str(end_idx) + '_' + str(begin_idx)

		bond_TypeAtt = []
		bond_TypeAtt = bond_TypeAtt + list(oneHotVector_with_fill(bond_type, bondtype_list_order))
		edge.TypeAtt = np.array(bond_TypeAtt, dtype=att_dtype)
		edge.connects = (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
		graph.edges.append(edge)
	# Add atoms
	for k, atom in enumerate(rdmol.GetAtoms()):
		node = Node()
		node.i = atom.GetIdx()
		node.attributes = atomAttributes(atom, selected_atom_list = selected_atom_list, extra_attributes = attributes[k])
		node_type = atom.GetAtomicNum()
		if node_type not in selected_atom_list:
			#print(rdmol)
			#print(Chem.MolToSmiles(rdmol))
			#print(node_type)
			raise AtomError("Atom type is not common, ignore the mol to prevent overfitting")
			#pass
		try:
			temp1 = atom.GetProp('atom_subtype')
		except KeyError:
			#print(atom.GetAtomicNum())
			raise SubtypeError
		temp = np.array(list(oneHotVector(temp1, subtype_list))) # or np.fromiter(map_object)
		node.subtype = np.argmax(temp) + 1 #to indicate the index of subtype.
		# this was for type att in 1st relational tensor. not useful anymore.
		node_TypeAtt = []
		node_TypeAtt = node_TypeAtt + list(oneHotVector_with_fill(node_type, atomtype_list_order))
		node.TypeAtt = np.array(node_TypeAtt, dtype=att_dtype)
		# this will return neighbor node index and bond index within the molecule for each atom.
		for neighbor in atom.GetNeighbors():
			node.neighbors.append((
				neighbor.GetIdx(),
				rdmol.GetBondBetweenAtoms(
					atom.GetIdx(),
					neighbor.GetIdx()
				).GetIdx()
			))
		graph.nodes.append(node)
	# Add counts, for convenience
	graph.num_edges = len(graph.edges)
	graph.num_nodes = len(graph.nodes)
	return graph 
"""
def bondAttributes(bond):
	'''Returns a numpy array of attributes for an RDKit bond

	From Neural FP defaults:
	The bond features were a concatenation of whether the bond type was single, double, triple,
	or aromatic, whether the bond was conjugated, and whether the bond was part of a ring.
	'''
	# Initialize
	attributes = []
	# Add bond type
	attributes += oneHotVector(
		bond.GetBondTypeAsDouble(),
		[1.0, 1.5, 2.0, 3.0]
	)
	# Add if is aromatic
	attributes.append(bond.GetIsAromatic())
	# Add if bond is conjugated
	attributes.append(bond.GetIsConjugated())
	# Add if bond is part of ring
	attributes.append(bond.IsInRing())

	# NEED THIS FOR TENSOR REPRESENTATION - 1 IF THERE IS A BOND
	attributes.append(1)

	return np.array(attributes, dtype = att_dtype)
"""
def atomAttributes(atom, selected_atom_list = [5, 6, 7, 8, 9, 15, 16, 17, 35, 53],extra_attributes = []):
	'''Returns a numpy array of attributes for an RDKit atom

	From ECFP defaults:
	<IdentifierConfiguration>
        <Property Name="AtomicNumber" Value="1"/>
        <Property Name="HeavyNeighborCount" Value="1"/>
        <Property Name="HCount" Value="1"/>
        <Property Name="FormalCharge" Value="1"/>
        <Property Name="IsRingAtom" Value="1"/>
    </IdentifierConfiguration>
    '''
	# Initialize
	if atom.GetAtomicNum() not in selected_atom_list:
		raise AtomError()
	attributes = []
	# Add atomic number (todo: finish)
	attributes += oneHotVector(
		atom.GetAtomicNum(),
		selected_atom_list
	)
	# Add heavy neighbor count
	if len(atom.GetNeighbors()) not in [0, 1, 2, 3, 4, 5]:
		print(atom.atomname, atom.GetAtomicNum)
	attributes += oneHotVector(
		len(atom.GetNeighbors()),
		[0, 1, 2, 3, 4, 5]
	)
	# Add hydrogen count
	attributes += oneHotVector(
		atom.GetTotalNumHs(),
		[0, 1, 2, 3, 4]
	)
	# Add formal charge
	attributes.append(atom.GetFormalCharge())
	# Add boolean if in ring
	attributes.append(atom.IsInRing())
	# Add boolean if aromatic atom
	attributes.append(atom.GetIsAromatic())

	attributes += extra_attributes

	return np.array(attributes, dtype = att_dtype)

def oneHotVector(val, lst):
	'''Converts a value to a one-hot vector based on options in lst'''
	if val not in lst:
		raise OneHotError()
		val = lst[-1]
	return map(lambda x: x == val, lst)

def oneHotVector_with_fill(val, lst):
	'''Converts a value to a one-hot vector based on options in lst'''
	if val not in lst:
		#raise OneHotError()
		val = lst[-1]
	return map(lambda x: x == val, lst)

# revised here to reduce dimention of bond-type attention tensor.
def sizeAttributeVectorsAtt(bondtype_list_order, atomtype_list_order, molecular_attributes = False):
	m = AllChem.MolFromSmiles('CC')
	g = molToGraph(m, bondtype_list_order,atomtype_list_order, molecular_attributes = molecular_attributes)
	a = g.nodes[0]
	b = g.edges[0]
	#return len(a.attributes), len(b.TypeAtt)+len(a.TypeAtt)
	return len(a.attributes), len(b.TypeAtt)


def fillBondType_dic(rdmol, bondtype_dic, selected_atom_list = [5, 6, 7, 8, 9, 15, 16, 17, 35, 53]):
	# Add bonds
	for bond in rdmol.GetBonds():
		BeginAtom, EndAtom = bond.GetBeginAtom(), bond.GetEndAtom()
		begin_idx, end_idx = BeginAtom.GetAtomicNum(), EndAtom.GetAtomicNum()

		if (begin_idx not in selected_atom_list) or (end_idx not in selected_atom_list):
			raise AtomError
		if begin_idx < end_idx:
			bond_type = str(begin_idx) + '_' + str(end_idx)
		else:
			bond_type = str(end_idx) + '_' + str(begin_idx)
		if bond_type in bondtype_dic.keys():
			bondtype_dic[bond_type] += 1
		else:
			bondtype_dic[bond_type] = 1
		if bond_type == '7_15' or bond_type == '6_15' or bond_type == '5_7':
			#print(bond_type, Chem.MolToSmiles(rdmol))
			pass
	return(bondtype_dic)

def fillAtomType_dic(rdmol, atomtype_dic, selected_atom_list = [5, 6, 7, 8, 9, 15, 16, 17, 35, 53]):
	for atom in rdmol.GetAtoms():
		atom_num = atom.GetAtomicNum()
		if atom_num not in selected_atom_list:
			raise AtomError
		if atom_num in atomtype_dic:
			atomtype_dic[atom_num] += 1
		else:
			atomtype_dic[atom_num] = 1
	return(atomtype_dic)

def sub_type_setup(m):
	for atom in m.GetAtoms():
		# print(atom.GetIdx(),  atom.GetSymbol(),
		#      atom.GetNumImplicitHs(), atom.GetIsAromatic(), atom.GetAtomicNum(), atom.GetHybridization(),
		#      atom.GetDegree(), atom.GetTotalDegree(), atom.GetExplicitValence(), atom.GetImplicitValence(),
		#      atom.GetNoImplicit(),atom.GetNumExplicitHs(), atom.GetNumImplicitHs(), atom.GetSmarts() )

		# atom types for oxygen
		if atom.GetAtomicNum() == 8:
			# print(atom.GetIdx(), atom.GetSymbol(), atom.GetHybridization(), int(atom.GetHybridization()),atom.GetDegree() )
			if atom.GetHybridization() == Chem.HybridizationType.SP3:  # SP3 O hybridization check
				if atom.GetDegree() == 2:  # alphatic C-(SP3 O)- alphatic C
					atom.SetProp("atom_subtype", "os")
				elif atom.GetDegree() == 1:  # -OH
					atom.SetProp("atom_subtype", "oh")
				else:
					atom.SetProp("atom_subtype", "un O")
					# print('unassigned O atom type')
			elif atom.GetHybridization() == Chem.HybridizationType.SP2:  ## probably still mess up the COO/C=O/aromatic O/aromatic C-O-Aromatic C
				# print(atom.GetIdx(), atom.GetSymbol(), atom.GetHybridization(), atom.GetDegree())
				if atom.GetIsAromatic():
					atom.SetProp("atom_subtype", "oa")  # o in the aromatic ring
				elif atom.GetDegree() == 1:
					atom.SetProp("atom_subtype", "o")  # COO
				elif atom.GetDegree() == 2:
					atom.SetProp("atom_subtype", "oa")  # temporary assign aromatic C-O-C to oa type
			else:
				# print(atom.GetIdx(), atom.GetSymbol(), atom.GetHybridization(), atom.GetDegree())
				atom.SetProp("atom_subtype", "un O")
				# print('unassigned O atom type')

			# atom types for carbon
		elif atom.GetAtomicNum() == 6:
			#print(atom.GetIdx(), atom.GetSymbol(), atom.GetHybridization(), int(atom.GetHybridization()),
			#	  atom.GetDegree())
			if atom.GetHybridization() == Chem.HybridizationType.SP3:  # SP3 C hybridization
				#print(atom.GetIdx(), atom.GetSymbol(), atom.GetHybridization(), atom.GetDegree())
				atom.SetProp('atom_subtype', 'c3')
			elif atom.GetHybridization() == Chem.HybridizationType.SP:  # SP C hydridization
				#print(atom.GetIdx(), atom.GetSymbol(), atom.GetHybridization(), atom.GetDegree())
				atom.SetProp('atom_subtype', 'c1')
			elif atom.GetHybridization() == Chem.HybridizationType.SP2:  # SP2 C hybridization
				if atom.GetIsAromatic():
					#print(atom.GetIdx(), atom.GetSymbol(), atom.GetHybridization(), atom.GetDegree())
					atom.SetProp('atom_subtype', 'ca')
				# elif atom.HasSubstructMatch('[CD3H0](=*)(-*)-*'): or elif atom.match('[CD3H0](=*)(-*)-*')
				# elif atom.Match('[CD3H0](=*)(-*)-*'):
				elif atom.GetNumImplicitHs() == 0 and atom.GetDegree() == 3:
					#print(atom.GetIdx(), atom.GetSymbol(), atom.GetHybridization(), atom.GetDegree())
					atom.SetProp('atom_subtype', 'c')  # very dirty way for CC(=O)C or C-C(=S)C
				elif atom.GetNumImplicitHs() == 1 and atom.GetDegree() == 2:
					atom.SetProp('atom_subtype', 'c2')  # very dirty way for C-C=C-C
					#print(atom.GetProp('atom_subtype'))
				else:
					# print('unassigned C atom type')
					atom.SetProp('atom_subtype', 'un C')
			else:
				# print(atom.GetIdx(), atom.GetSymbol(), atom.GetHybridization(), atom.GetDegree())
				atom.SetProp('atom_subtype', 'un C')
				# print('unassigned C atom type')

			# atom types for nitrogen
		elif atom.GetAtomicNum() == 7:
			if atom.GetHybridization() == Chem.HybridizationType.SP:  # SP N hydridization
				atom.SetProp('atom_subtype', 'n1')
			elif atom.GetHybridization() == Chem.HybridizationType.SP3:  # SP3 N hybridization
				atom.SetProp('atom_subtype', 'n3')
			elif atom.GetHybridization() == Chem.HybridizationType.SP2:  # SP2 N hybridization
				if atom.GetIsAromatic():
					atom.SetProp('atom_subtype', 'na')
				elif atom.GetDegree() == 3:
					if atom.GetExplicitValence() == 3:  # amide N or N next to aromatic ring, but no H
						atom.SetProp('atom_subtype', 'n')
					elif atom.GetExplicitValence() == 4 and atom.GetNumImplicitHs() == 2:  # nitro amide
						atom.SetProp('atom_subtype', 'no')
					else:
						atom.SetProp('atom_subtype', 'un N')
						# print('unassigned N atom type')
				elif atom.GetDegree() == 1:  # -NH2 in the adenine ring
					atom.SetProp('atom_subtype', 'n')  # temportarily assigned to n here
				elif atom.GetDegree() == 2:  # amide N or next to aromatic ring, but with one H substitution
					atom.SetProp('atom_subtype', 'nh')
				else:
					atom.SetProp('atom_subtype', 'un N')
					# print('unassigned N atom type')
			else:
				# print(atom.GetIdx(), atom.GetSymbol(), atom.GetHybridization(), atom.GetDegree())
				atom.SetProp('atom_subtype', 'un N')
				# print('unassigned N atom type')

			# atom types for sulfur
		elif atom.GetAtomicNum() == 16:
			#print(atom.GetIdx(), atom.GetSymbol(), atom.GetHybridization(), atom.GetDegree())
			if atom.GetHybridization() == Chem.HybridizationType.SP2:
				# print(atom.GetIdx(), atom.GetSymbol(), atom.GetHybridization(), atom.GetDegree())
				atom.SetProp('atom_subtype', 's2')  # -C=S
			elif atom.GetHybridization() == Chem.HybridizationType.SP3:
				if atom.GetDegree() == 1 and atom.GetNumImplicitHs() == 1:  # -SH
					# print(atom.GetIdx(), atom.GetSymbol(), atom.GetHybridization(), atom.GetDegree())
					atom.SetProp('atom_subtype', '-sh')
				elif atom.GetDegree() == 2:  # do not diffentiate -S-S- and C-S-C currently
					# print(atom.GetIdx(), atom.GetSymbol(), atom.GetHybridization(), atom.GetDegree())
					atom.SetProp('atom_subtype', '-ss')
				else:
					atom.SetProp('atom_subtype', 'un S')
					# print('unassigned S atom type')
			else:
				# print(atom.GetIdx(), atom.GetSymbol(), atom.GetHybridization(), atom.GetDegree())
				atom.SetProp('atom_subtype', 'un S')
				# print('unassigned S atom type')

			# atom types for halogens (F, Cl, Br)
		elif atom.GetAtomicNum() == 9:
			atom.SetProp('atom_subtype', 'f')
		elif atom.GetAtomicNum() == 17:
			atom.SetProp('atom_subtype', 'cl')
		elif atom.GetAtomicNum() == 35:
			atom.SetProp('atom_subtype', 'br')
		elif atom.GetAtomicNum() == 53:
			atom.SetProp('atom_subtype', 'i')

		# atom types for phosphorus; only SP3 P and do not inplement SP2 C=P here
		elif atom.GetAtomicNum() == 15 & atom.GetHybridization() == Chem.HybridizationType.SP3:
			atom.SetProp('atom_subtype', 'p')

		else:
			atom.SetProp('atom_subtype', 'unreconginized')
			# print('unreconginized atom types {}'.format(atom.GetAtomicNum()))

	return m


class GraphError(BaseException):
	pass

class AtomError(BaseException):
	pass

class OneHotError(BaseException):
	pass

class SubtypeError(BaseException):
	pass