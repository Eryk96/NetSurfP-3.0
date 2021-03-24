
"""
mmCIF parsing
"""

import re

FIELDS_PATTERN = re.compile(r"(?:\'(.*?)\'|(\S+))")


def regex_split(data):
    return [x[0] or x[1] for x in FIELDS_PATTERN.findall(data)]


import collections

class mmCIF:

    def __init__(self):
        self.records = collections.OrderedDict()
        self.categories = collections.OrderedDict()
        self.datablock = None

        self.auth2label = {}
        self.label2auth = {}

    @classmethod
    def parse(cls, filehandle):
        """Rudimentary parsing of mmCIF files."""
        parsed = cls()
        
        def _parse_multiline(line):
            if line.startswith(';'):
                fields = [line[1:].strip()]
                line = next(filehandle)
                while not line.endswith(';'):
                    fields[0] += line.strip()
                    line = next(filehandle).strip()
                fields[0] += line.strip(';')
            else:
                fields = regex_split(line)

            return fields

        for line in filehandle:
            if line.startswith('data_'):
                parsed.datablock = line.strip()
            elif line.lstrip().startswith('#'):
                pass
            elif line.startswith('_'):
                fields = regex_split(line)
                if len(fields) == 1:
                    fields.extend(_parse_multiline(next(filehandle)))
                assert len(fields) == 2, line
                parsed.add_single_item(*fields)
            #
            elif line.startswith('loop_'):
                loop_headers = []
                for line in filehandle:
                    if line.startswith('#'):
                        break
                    elif line.startswith('_'):
                        line = line.strip()
                        parsed.add_list_item(line)
                        loop_headers.append(line)
                    else:
                        line = _parse_multiline(line)
                        while len(loop_headers) != len(line):
                            line.extend(_parse_multiline(next(filehandle)))

                        assert len(loop_headers) == len(line), (line, loop_headers)
                        row = {key: val for key, val in zip(loop_headers, line)}
                        parsed.append_row(row)

                        #Build chain ID translation table
                        if '_atom_site.group_PDB' in row and row['_atom_site.group_PDB'] == 'ATOM':
                            label_id = row['_atom_site.label_asym_id']
                            auth_id  = row['_atom_site.auth_asym_id']
                            if auth_id in parsed.auth2label:
                                assert parsed.auth2label[auth_id] == label_id, (parsed.auth2label[auth_id], label_id)
                            else:
                                parsed.auth2label[auth_id] = label_id
                                parsed.label2auth[label_id] = auth_id

        return parsed

    #
    # Parsing stuff
    #

    def add_single_item(self, key, val):
        self.records[key] = str(val)
        
        category = key.split('.')[0]
        if category not in self.categories:
            self.categories[category] = []

        self.categories[category].append(key)

    def add_list_item(self, key):
        self.records[key] = []
        
        category = key.split('.')[0]
        if category not in self.categories:
            self.categories[category] = []

        self.categories[category].append(key)

    def append_item(self, key, val):
        self.records[key].append(val)

    def append_row(self, row):
        for key, val in row.items():
            self.records[key].append(str(val))

    #
    # Manipulate mmCIF
    #

    def isolate_chains(self, label_asym_ids):
        """`asym_ids` is the mmCIF chain ID, NOT the normal PDB chain ID."""
        import copy

        new_cif = copy.deepcopy(self)

        atom_keys = self.categories['_atom_site']
        for k in atom_keys:
            new_cif.add_list_item(k)

        label_key = '_atom_site.label_asym_id'
        for i, label_asym_id in enumerate(self.records[label_key]):
            if label_asym_id in label_asym_ids:
                for key in atom_keys:
                    new_cif.append_item(key, self.records[key][i])

        return new_cif

    def write(self, filehandle):
        touched = set()

        print(self.datablock, file=filehandle)
        print('#', file=filehandle)

        for cat, keys in self.categories.items():
            if isinstance(self.records[keys[0]], str):
                for key in keys:
                    val = self.records[key]
                    if ' ' in val:
                        val = "'{}'".format(val)
                    print(key, val, '', file=filehandle)
            else:
                print('loop_', file=filehandle)
                for key in keys:
                    print(key, '', file=filehandle)
                for vals in zip(*[self.records[key] for key in keys]):
                    for val in vals:
                        if ' ' in val:
                            val = "'{}'".format(val)
                        print(val, end=' ', file=filehandle)
                    print(file=filehandle)

            print('#', file=filehandle)
