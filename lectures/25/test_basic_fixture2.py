
import pytest
def test_comparewithAA_file2(supply_AA_BB_CC):
	zz=25
	assert supply_AA_BB_CC[0]==zz,"aa and zz comparison failed"

def test_comparewithBB_file2(supply_AA_BB_CC):
	zz=25
	assert supply_AA_BB_CC[1]==zz,"bb and zz comparison failed"

def test_comparewithCC_file2(supply_AA_BB_CC):
	zz=25
	assert supply_AA_BB_CC[2]==zz,"cc and zz comparison failed"
