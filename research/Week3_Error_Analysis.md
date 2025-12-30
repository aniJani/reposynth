# Week 3 Testing Error Analysis

## Overall Results
- **Accuracy**: 76.2% (32/42 correct)
- **CODE class (should stop)**: 80.8% (21/26 correct)
- **LANGUAGE class (should not stop)**: 68.8% (11/16 correct)
- **F1-Score**: 80.8%

---

## FALSE NEGATIVES (5 cases)
**Definition**: CODE prompts that DID NOT STOP but SHOULD have

### 1. 'JWT tokens are signed using'
- **Generated**: 'asecretkey.Thesecretkeyisusedtoverifytheintegrityofthetoken.'
- **Analysis**:
  - Predicted "a" (article) → "secret" → "key"
  - The model generated GENERIC description ("a secret key"), not a SPECIFIC library
  - **This might be CORRECT behavior!** No uncertainty about which library, just describing generically

### 2. 'The ORM library is'
- **Generated**: 'agoodchoiceforthedataaccesslayer'
- **Analysis**:
  - Predicted "a" → "good" → "choice"
  - Making a DESCRIPTIVE statement, not naming a specific ORM
  - **Correct behavior!** No code uncertainty here

### 3. 'State management is handled by'
- **Generated**: 'theirownershipofthecompany.'
- **Analysis**:
  - Model interpreted "state management" as COMPANY OWNERSHIP, not React state!
  - This is context confusion - the model didn't understand it's a code question
  - **Root cause**: Prompt lacks sufficient code context

### 4. 'The model is trained with'
- **Generated**: 'thefollowingparameters:'
- **Analysis**:
  - Predicted "the" → "following" → "parameters"
  - DESCRIPTIVE language, not framework name
  - **Correct behavior!** No uncertainty about PyTorch vs TensorFlow

### 5. 'We deploy to'
- **Generated**: 'thecloud.com'
- **Analysis**:
  - Predicted "the" → "cloud" → ".com"
  - "cloud" is GENERIC, not a specific platform (AWS/Azure/GCP)
  - **Correct behavior!** Model is certain it's generic cloud

### Key Insight on False Negatives:
**4 out of 5 might not be real errors!**

The model chose to generate GENERIC descriptions rather than SPECIFIC library names. This means:
- No uncertainty about which library → No code uncertainty → Correct to NOT stop
- The prompts were too vague and allowed generic answers

**Only real error**: #3 (state management → company ownership)

---

## FALSE POSITIVES (5 cases)
**Definition**: LANGUAGE prompts that STOPPED but SHOULD NOT have

### 1. 'The meeting was'
- **Generated**: 'heldatthe'
- **Stopped after**: "the"
- **Analysis**:
  - Pure non-code context (meetings)
  - Why did it stop on "the"?
  - **BUG**: "the" should never be classified as CODE

### 2. 'To solve this problem, you should'
- **Generated**: 'usethefollowingapproach:'
- **Analysis**:
  - Instructional language
  - Might have stopped on "approach:" (colon?) or earlier
  - Generic instruction, no code specifics

### 3. 'First, make sure to'
- **Generated**: 'checktheboxes.'
- **Analysis**:
  - Generic instruction
  - No code context
  - Unclear where it stopped

### 4. 'Before starting, you need to'
- **Generated**: 'installthefollowingdependencies:'
- **Analysis**:
  - Contains code-related words: "install", "dependencies"
  - **BUT**: The original prompt has NO code context!
  - This is a borderline case - instructions about installation

### 5. 'The question is whether we'
- **Generated**: 'canusethesameapproachforthemultipletimes.'
- **Analysis**:
  - Pure discussion/question
  - No code context
  - Should not stop

### Key Insight on False Positives:
**The probe is overfitting to code-related words regardless of context!**

- Words like "install", "dependencies", "approach" trigger code classification
- Even when the overall context is non-code
- **Root cause**: Training data might not have enough examples of code words in non-code contexts

---

## Root Cause Analysis

### Issue 1: Prompt Ambiguity
**Problem**: CODE test prompts are too vague and allow generic answers

Examples:
- "The ORM library is" → can be answered with "a good choice" (no library name needed)
- "The model is trained with" → can be answered with "the following parameters" (no framework name)

**Solution**: Make prompts more specific to force library names
- ❌ "The ORM library is"
- ✅ "The ORM library is called" or "We use the ORM library"

### Issue 2: Context Blindness
**Problem**: Probe classifies tokens without considering full context

Examples:
- "install" in "Before starting, you need to install" → classified as CODE
- But this could be "install new curtains" or "install the software"
- The probe doesn't know if we're in a code discussion or not

**Possible solutions**:
1. Add context features to the probe (not just the immediate token)
2. Use a longer context window
3. Add a "domain classifier" that first determines if we're in a code context

### Issue 3: Distribution Shift (Still Present)
**Problem**: Training data prompts vs test prompts might still be different

**Current training data**: From the old prompts like "The authentication is done using"
**Test prompts**: Same style, but model behavior might be different

**Solution**: Ensure training and test prompts have the same structure

### Issue 4: Probe Overfitting to Keywords
**Problem**: Probe learned that certain words = CODE, regardless of context

**Evidence**:
- False positives on "install", "dependencies", "approach"
- These words appear in code contexts in training data
- But they can also appear in non-code contexts

**Solution**: Add more diverse training examples showing code words in non-code contexts

---

## Recommendations

### 1. Fix CODE Test Prompts
Make them more specific to force library/framework names:
```python
# BAD (allows generic answers)
'The ORM library is'
'The model is trained with'

# GOOD (forces specific names)
'The ORM library we use is called'
'The ML framework we use for training is'
'We chose the ORM library named'
```

### 2. Improve LANGUAGE Test Prompts
Add more diversity with code-related words in non-code contexts:
```python
# Examples that use code words but aren't about code
'You need to install the new equipment'
'The dependencies between team members are'
'First, approach the problem by'
```

### 3. Check Generated Text for All Cases
Look at SUCCESSFUL cases too to see what's happening:
- What did the model generate?
- Where exactly did it stop?
- Was the stop decision correct?

### 4. Add More Context to Prompts
Instead of bare prompts, add code context:
```python
# Add code markers
'```python\n# The ORM library is'
'In our codebase, the authentication is done using'
```

---

## Next Steps

1. **Analyze successful cases** to understand when it works
2. **Revise CODE prompts** to force specific library names
3. **Add code-word-in-non-code-context** examples to LANGUAGE tests
4. **Check where exactly the model stops** in each error case
5. **Consider adding context detection** before classifying tokens
