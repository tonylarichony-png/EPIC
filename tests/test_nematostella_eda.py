"""Small independent checks for genomic indexing and source-file auditing."""
import gzip
import importlib.util
import io
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

SPEC = importlib.util.spec_from_file_location("nematostella_eda", Path(__file__).parents[1] / "scripts/eda_nematostella.py")
eda = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(eda)


class TestNematostellaEDA(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.seqs = {"c1":eda.LUT[np.frombuffer(b"ACGT",dtype=np.uint8)],
                     "c2":eda.LUT[np.frombuffer(b"AA",dtype=np.uint8)]}
        self.w = eda.Whitelist([("c1",1,4)],self.seqs)

    def tearDown(self):
        self.tmp.cleanup()

    def gz(self,name,text):
        path=self.root/name
        with gzip.open(path,"wt",encoding="ascii") as f: f.write(text)
        return path

    def test_strands_and_interval_expansion(self):
        path=self.gz("signal.bed.gz","c1\t1\t3\tx\t2\t+\nc1\t3\t4\ty\t3\t-\n")
        ix,val,report=eda.read_bed(path,self.w)
        np.testing.assert_array_equal(ix,[0,1,5])
        np.testing.assert_array_equal(val,[2,2,3])
        self.assertEqual(report["read_sum"],7)
        names,positions,minus=self.w.decode(np.arange(6))
        np.testing.assert_array_equal(positions,[1,2,3,1,2,3])
        np.testing.assert_array_equal(minus,[False,False,False,True,True,True])
        actual=eda.oriented_context(self.seqs,self.w.decode(np.array([0,3])),[-1,0,1])
        np.testing.assert_array_equal(actual,[[0,1,2],[1,2,3]])

    def test_context_never_crosses_contig_boundary(self):
        w=eda.Whitelist([("c2",0,2)],self.seqs)
        actual=eda.oriented_context(self.seqs,w.decode(np.array([0,3])),[-1,0,1])
        np.testing.assert_array_equal(actual,[[4,0,0],[4,3,3]])

    def test_dense_txt_all_values_and_chunk_boundaries(self):
        path=self.gz("target.txt.gz","2\n2\n0\n0\n0\n3\n")
        report=eda.audit_dense_txt(path,np.array([0,1,5]),np.array([2,2,3]),6,block_bytes=3)
        self.assertEqual(report,{"lines":6,"nonzero":3,"read_sum":7,"mismatches":0})
        float_path=self.gz("float.txt.gz","2.0\n2e0\n0.0\n0\n0\n3.000\n")
        self.assertEqual(eda.audit_dense_txt(float_path,np.array([0,1,5]),np.array([2,2,3]),6,block_bytes=3),report)
        fractional=self.gz("fractional.txt.gz","2.5\n")
        with self.assertRaises(ValueError):
            eda.audit_dense_txt(fractional,np.array([0]),np.array([2]),1)
        with self.assertRaises(ValueError):
            eda.audit_dense_txt(path,np.array([0,1,4]),np.array([2,2,3]),6,block_bytes=3)
        with self.assertRaises(ValueError):
            eda.audit_dense_txt(path,np.array([0,1,5]),np.array([2,2,4]),6,block_bytes=3)
        with self.assertRaises(ValueError):
            eda.audit_dense_txt(path,np.array([0,1,5]),np.array([2,2,3]),7)

    def test_overlap_duplicate_and_outside_keys_rejected(self):
        with self.assertRaises(ValueError):
            eda.Whitelist([("c1",0,3),("c1",2,4)],self.seqs)
        with self.assertRaises(ValueError):
            self.w.encode_interval("c1",0,1,"+")
        with self.assertRaises(ValueError):
            self.w.encode_interval("c1",1,5,"+")
        path=self.gz("dupe.bed.gz","c1\t1\t2\tx\t2\t+\nc1\t1\t2\tx\t2\t+\n")
        with self.assertRaises(ValueError): eda.read_bed(path,self.w)

    def test_template_and_dinucleotides(self):
        path=self.gz("template.bed.gz","c1\t1\t4\t.\t0\t+\nc1\t1\t4\t.\t0\t-\n")
        self.w.audit_template(path)
        ctx=eda.oriented_context(self.seqs,self.w.decode(np.arange(6)),[-1,0])
        expected=np.zeros(17,dtype=np.int64)
        for a,b in ctx:
            expected[int(a)*4+int(b) if a<4 and b<4 else 16]+=1
        np.testing.assert_array_equal(eda.full_dinucleotides(self.seqs,self.w),expected)
        wrong=self.gz("wrong.bed.gz","c1\t1\t4\t.\t0\t-\nc1\t1\t4\t.\t0\t+\n")
        with self.assertRaises(ValueError): self.w.audit_template(wrong)

    def test_replicate_sum(self):
        ix,total,a,b=eda.merge_tracks((np.array([0,2]),np.array([1,4])),(np.array([1,2]),np.array([2,3])))
        np.testing.assert_array_equal(ix,[0,1,2])
        np.testing.assert_array_equal(total,[1,2,7])

    def test_hidden_answers_not_extracted(self):
        archive=self.root/"source.tgz"
        with tarfile.open(archive,"w:gz") as ar:
            for name in ["genome/test.txt","ground_truth/csRNA-test.txt.gz","scoring/mask.txt"]:
                entry=tarfile.TarInfo(eda.ASM+"/"+name)
                payload=b"example"
                entry.size=len(payload)
                ar.addfile(entry,io.BytesIO(payload))
        with patch.object(eda,"MD5",eda.digest(archive,"md5")):
            root,files=eda.extract_inputs(archive,self.root/"unpacked")
        self.assertEqual(len(files),1)
        self.assertTrue((root/"genome/test.txt").exists())
        self.assertFalse((root/"ground_truth").exists())
        self.assertFalse((root/"scoring").exists())


if __name__ == "__main__":
    unittest.main()
