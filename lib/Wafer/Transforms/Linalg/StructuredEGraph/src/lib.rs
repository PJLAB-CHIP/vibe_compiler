use egg::{
    Analysis, Applier, CostFunction, DidMerge, EGraph, Extractor, Id, Language, PatternAst,
    Rewrite, RewriteScheduler, Runner, SearchMatches, Searcher, StopReason, Subst, Symbol, Var,
};
use std::cell::Cell;
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::ffi::c_void;
use std::fmt::{self, Debug, Display, Formatter};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::rc::Rc;
use std::slice;
use std::sync::{
    atomic::{AtomicU32, AtomicU64, Ordering as AtomicOrdering},
    Arc, Mutex,
};
use std::time::Duration;

const SCHEMA_VERSION: u32 = 2;
const FLAG_FORCE_PANIC: u32 = 1;

const NODE_INPUT: u32 = 1;
const NODE_ACCESS: u32 = 2;
const NODE_ELEMENTWISE: u32 = 3;
const NODE_CONTRACTION: u32 = 4;
const NODE_REDUCTION: u32 = 5;
const NODE_CONCAT: u32 = 6;

const STATUS_CHANGED: u32 = 0;
const STATUS_UNCHANGED: u32 = 1;
const STATUS_BUDGET_EXHAUSTED: u32 = 2;
const STATUS_INVALID_INPUT: u32 = 3;
const STATUS_INTERNAL_ERROR: u32 = 4;

const CALLBACK_EXACT: u32 = 0;
const CALLBACK_UNSUPPORTED: u32 = 1;
const CALLBACK_WORK_LIMIT: u32 = 2;
const CALLBACK_INTERNAL_ERROR: u32 = 3;

const RELATION_IDENTITY: u32 = 1 << 0;
const RELATION_MATERIALIZABLE: u32 = 1 << 5;
const RELATION_CANONICAL_RESHAPE: u32 = 1 << 6;

const RUNTIME_RUNNING: u32 = 0;
const RUNTIME_BUDGET: u32 = 1;
const RUNTIME_INTERNAL_ERROR: u32 = 2;
const MAX_CONCAT_ACCESS_COMBINATIONS: usize = 64;

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct WaferEGraphNode {
    kind: u32,
    semantic_id: u32,
    type_id: u32,
    child_offset: u32,
    child_count: u32,
    relation_offset: u32,
    relation_count: u32,
    data_input_count: u32,
    axis: i32,
    source_order: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct WaferEGraphRelationFacts {
    destination_type_id: u32,
    source_type_id: u32,
    flags: u32,
    reserved: u32,
}

type GetRelationFactsFn =
    unsafe extern "C" fn(*mut c_void, u32, *mut WaferEGraphRelationFacts) -> u32;
type ComposeRelationsFn = unsafe extern "C" fn(*mut c_void, u32, u32, *mut u32) -> u32;
type ValidateComputeFn =
    unsafe extern "C" fn(*mut c_void, u32, u32, u32, *const u32, *const u32, u64, u32) -> u32;
type ReindexComputeFn = unsafe extern "C" fn(
    *mut c_void,
    u32,
    u32,
    u32,
    u32,
    u32,
    *const u32,
    u64,
    u32,
    *mut u32,
    *mut u32,
) -> u32;
type ReparameterizeElementwiseFn = unsafe extern "C" fn(
    *mut c_void,
    u32,
    u32,
    u32,
    u32,
    *const u32,
    u64,
    u32,
    *mut u32,
    *mut u32,
    *mut u32,
) -> u32;
type ReparameterizeReshapeComputeFn = unsafe extern "C" fn(
    *mut c_void,
    u32,
    u32,
    u32,
    u32,
    u32,
    *const u32,
    u64,
    u32,
    *mut u32,
    *mut u32,
    *mut u32,
    *mut u32,
    *mut u32,
) -> u32;
type ValidateConcatFn = unsafe extern "C" fn(*mut c_void, u32, i32, *const u32, u64) -> u32;
type FactorConcatFn = unsafe extern "C" fn(
    *mut c_void,
    u32,
    i32,
    *const u32,
    *const u32,
    u64,
    *mut u32,
    *mut u32,
    *mut i32,
) -> u32;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct WaferEGraphRelationService {
    context: *mut c_void,
    get_relation_facts: Option<GetRelationFactsFn>,
    compose_relations: Option<ComposeRelationsFn>,
    validate_compute: Option<ValidateComputeFn>,
    reindex_compute: Option<ReindexComputeFn>,
    reparameterize_elementwise: Option<ReparameterizeElementwiseFn>,
    reparameterize_reshape_compute: Option<ReparameterizeReshapeComputeFn>,
    validate_concat: Option<ValidateConcatFn>,
    factor_concat: Option<FactorConcatFn>,
}

#[repr(C)]
pub struct WaferEGraphRequest {
    schema_version: u32,
    flags: u32,
    nodes: *const WaferEGraphNode,
    node_count: u64,
    children: *const u32,
    child_count: u64,
    relations: *const u32,
    relation_count: u64,
    root_nodes: *const u32,
    root_count: u64,
    maximum_iterations: u32,
    maximum_e_nodes: u64,
    maximum_matches: u64,
    relation_service: WaferEGraphRelationService,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct WaferEGraphStatistics {
    input_nodes: u64,
    input_relations: u64,
    relation_queries: u64,
    e_nodes: u64,
    e_classes: u64,
    rewrite_matches: u64,
    e_class_merges: u64,
    rebuild_work: u64,
    iterations: u64,
    extraction_work: u64,
    output_nodes: u64,
    input_compute_occurrences: u64,
    output_compute_occurrences: u64,
    input_access_occurrences: u64,
    output_access_occurrences: u64,
    input_concat_occurrences: u64,
    output_concat_occurrences: u64,
    identity_applications: u64,
    composition_applications: u64,
    compute_absorption_applications: u64,
    concat_applications: u64,
    result_reindex_applications: u64,
    reshape_through_compute_applications: u64,
    input_records: u64,
    output_records: u64,
    input_bytes: u64,
    output_bytes: u64,
    search_limit_reached: u64,
    application_checks: u64,
    unchanged_applications: u64,
}

#[repr(C)]
pub struct WaferEGraphResult {
    status: u32,
    reserved: u32,
    nodes: *mut WaferEGraphNode,
    node_count: u64,
    children: *mut u32,
    child_count: u64,
    relations: *mut u32,
    relation_count: u64,
    root_nodes: *mut u32,
    root_count: u64,
    statistics: WaferEGraphStatistics,
}

impl WaferEGraphResult {
    fn status(status: u32) -> Self {
        Self {
            status,
            reserved: 0,
            nodes: std::ptr::null_mut(),
            node_count: 0,
            children: std::ptr::null_mut(),
            child_count: 0,
            relations: std::ptr::null_mut(),
            relation_count: 0,
            root_nodes: std::ptr::null_mut(),
            root_count: 0,
            statistics: WaferEGraphStatistics::default(),
        }
    }
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
struct GraphLanguage {
    kind: u32,
    semantic_id: u32,
    type_id: u32,
    relations: Box<[u32]>,
    data_input_count: u32,
    axis: i32,
    source_order: u32,
    children: Box<[Id]>,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct GraphDiscriminant {
    kind: u32,
    semantic_id: u32,
    type_id: u32,
    relations: Box<[u32]>,
    data_input_count: u32,
    axis: i32,
    source_order: u32,
    arity: usize,
}

impl Language for GraphLanguage {
    type Discriminant = GraphDiscriminant;

    fn discriminant(&self) -> Self::Discriminant {
        GraphDiscriminant {
            kind: self.kind,
            semantic_id: self.semantic_id,
            type_id: self.type_id,
            relations: self.relations.clone(),
            data_input_count: self.data_input_count,
            axis: self.axis,
            source_order: self.source_order,
            arity: self.children.len(),
        }
    }

    fn matches(&self, other: &Self) -> bool {
        self.discriminant() == other.discriminant()
    }

    fn children(&self) -> &[Id] {
        &self.children
    }

    fn children_mut(&mut self) -> &mut [Id] {
        &mut self.children
    }
}

impl Display for GraphLanguage {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "k{}:s{}:t{}:r{:?}",
            self.kind, self.semantic_id, self.type_id, self.relations
        )
    }
}

fn is_compute_kind(kind: u32) -> bool {
    matches!(kind, NODE_ELEMENTWISE | NODE_CONTRACTION | NODE_REDUCTION)
}

fn valid_node_shape(node: &GraphLanguage) -> bool {
    match node.kind {
        NODE_INPUT => {
            node.semantic_id != 0
                && node.children.is_empty()
                && node.relations.is_empty()
                && node.data_input_count == 0
                && node.axis < 0
        }
        NODE_ACCESS => {
            node.semantic_id == 0
                && node.children.len() == 1
                && node.relations.len() == 1
                && node.data_input_count == 0
                && node.axis < 0
        }
        NODE_CONCAT => {
            node.semantic_id == 0
                && !node.children.is_empty()
                && node.relations.is_empty()
                && node.data_input_count == 0
                && node.axis >= 0
        }
        kind if is_compute_kind(kind) => {
            node.semantic_id != 0
                && !node.children.is_empty()
                && node.relations.len() == node.children.len()
                && node.data_input_count as usize <= node.children.len()
                && node.axis < 0
        }
        _ => false,
    }
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct ComputeValidationKey {
    semantic_id: u32,
    kind: u32,
    result_type: u32,
    relations: Vec<u32>,
    operand_types: Vec<u32>,
    data_input_count: u32,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct ReindexKey {
    semantic_id: u32,
    kind: u32,
    source_type: u32,
    destination_type: u32,
    result_relation: u32,
    input_relations: Vec<u32>,
    data_input_count: u32,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct ReshapeComputeKey {
    semantic_id: u32,
    kind: u32,
    result_type: u32,
    access_operand: u32,
    access_relation: u32,
    input_relations: Vec<u32>,
    data_input_count: u32,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct ConcatValidationKey {
    result_type: u32,
    axis: i32,
    input_types: Vec<u32>,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct ConcatFactorKey {
    result_type: u32,
    axis: i32,
    relations: Vec<u32>,
    source_types: Vec<u32>,
}

type ReparameterizationValue = (Vec<u32>, Vec<u32>, Vec<u32>);
type ReshapeComputeValue = (u32, u32, Vec<u32>, Vec<u32>, Vec<u32>);

#[derive(Default)]
struct RuntimeState {
    status: AtomicU32,
    relation_queries: AtomicU64,
    identity_applications: AtomicU64,
    composition_applications: AtomicU64,
    compute_absorption_applications: AtomicU64,
    concat_applications: AtomicU64,
    result_reindex_applications: AtomicU64,
    reshape_through_compute_applications: AtomicU64,
    application_checks: AtomicU64,
    unchanged_applications: AtomicU64,
    relation_facts_cache: Mutex<BTreeMap<u32, Option<WaferEGraphRelationFacts>>>,
    composition_cache: Mutex<BTreeMap<(u32, u32), Option<u32>>>,
    compute_validation_cache: Mutex<BTreeMap<ComputeValidationKey, bool>>,
    reindex_cache: Mutex<BTreeMap<ReindexKey, Option<(Vec<u32>, bool)>>>,
    reparameterization_cache: Mutex<BTreeMap<ReindexKey, Option<ReparameterizationValue>>>,
    reshape_compute_cache: Mutex<BTreeMap<ReshapeComputeKey, Option<ReshapeComputeValue>>>,
    concat_validation_cache: Mutex<BTreeMap<ConcatValidationKey, bool>>,
    concat_factor_cache: Mutex<BTreeMap<ConcatFactorKey, Option<(u32, u32, i32)>>>,
}

impl RuntimeState {
    fn is_running(&self) -> bool {
        self.status.load(AtomicOrdering::Relaxed) == RUNTIME_RUNNING
    }

    fn status(&self) -> u32 {
        self.status.load(AtomicOrdering::Relaxed)
    }

    fn set_status(&self, status: u32) {
        self.status.store(status, AtomicOrdering::Relaxed);
    }

    fn increment(counter: &AtomicU64) {
        counter.fetch_add(1, AtomicOrdering::Relaxed);
    }

    fn record_callback_status(&self, status: u32) -> bool {
        Self::increment(&self.relation_queries);
        match status {
            CALLBACK_EXACT => true,
            CALLBACK_UNSUPPORTED => false,
            CALLBACK_WORK_LIMIT => {
                self.set_status(RUNTIME_BUDGET);
                false
            }
            CALLBACK_INTERNAL_ERROR => {
                self.set_status(RUNTIME_INTERNAL_ERROR);
                false
            }
            _ => {
                self.set_status(RUNTIME_INTERNAL_ERROR);
                false
            }
        }
    }
}

#[derive(Clone, Copy)]
struct RelationService {
    context: usize,
    get_relation_facts: GetRelationFactsFn,
    compose_relations: ComposeRelationsFn,
    validate_compute_fn: ValidateComputeFn,
    reindex_compute_fn: ReindexComputeFn,
    reparameterize_elementwise_fn: ReparameterizeElementwiseFn,
    reparameterize_reshape_compute_fn: ReparameterizeReshapeComputeFn,
    validate_concat_fn: ValidateConcatFn,
    factor_concat_fn: FactorConcatFn,
}

impl RelationService {
    fn from_raw(raw: WaferEGraphRelationService) -> Option<Self> {
        if raw.context.is_null() {
            return None;
        }
        Some(Self {
            context: raw.context as usize,
            get_relation_facts: raw.get_relation_facts?,
            compose_relations: raw.compose_relations?,
            validate_compute_fn: raw.validate_compute?,
            reindex_compute_fn: raw.reindex_compute?,
            reparameterize_elementwise_fn: raw.reparameterize_elementwise?,
            reparameterize_reshape_compute_fn: raw.reparameterize_reshape_compute?,
            validate_concat_fn: raw.validate_concat?,
            factor_concat_fn: raw.factor_concat?,
        })
    }

    fn context(self) -> *mut c_void {
        self.context as *mut c_void
    }

    fn facts(&self, relation_id: u32, runtime: &RuntimeState) -> Option<WaferEGraphRelationFacts> {
        if !runtime.is_running() || relation_id == 0 {
            return None;
        }
        match runtime.relation_facts_cache.lock() {
            Ok(cache) => {
                if let Some(cached) = cache.get(&relation_id) {
                    return *cached;
                }
            }
            Err(_) => {
                runtime.set_status(RUNTIME_INTERNAL_ERROR);
                return None;
            }
        }
        let mut facts = WaferEGraphRelationFacts::default();
        let status = unsafe { (self.get_relation_facts)(self.context(), relation_id, &mut facts) };
        let value = if runtime.record_callback_status(status)
            && facts.destination_type_id != 0
            && facts.source_type_id != 0
            && facts.reserved == 0
        {
            Some(facts)
        } else {
            if status == CALLBACK_EXACT {
                runtime.set_status(RUNTIME_INTERNAL_ERROR);
            }
            None
        };
        if runtime.is_running() {
            match runtime.relation_facts_cache.lock() {
                Ok(mut cache) => {
                    cache.insert(relation_id, value);
                }
                Err(_) => runtime.set_status(RUNTIME_INTERNAL_ERROR),
            }
        }
        value
    }

    fn compose(&self, outer: u32, inner: u32, runtime: &RuntimeState) -> Option<u32> {
        if !runtime.is_running() || outer == 0 || inner == 0 {
            return None;
        }
        let key = (outer, inner);
        match runtime.composition_cache.lock() {
            Ok(cache) => {
                if let Some(cached) = cache.get(&key) {
                    return *cached;
                }
            }
            Err(_) => {
                runtime.set_status(RUNTIME_INTERNAL_ERROR);
                return None;
            }
        }
        let mut result = 0;
        let status = unsafe { (self.compose_relations)(self.context(), outer, inner, &mut result) };
        let value = if runtime.record_callback_status(status) && result != 0 {
            Some(result)
        } else {
            if status == CALLBACK_EXACT {
                runtime.set_status(RUNTIME_INTERNAL_ERROR);
            }
            None
        };
        if runtime.is_running() {
            match runtime.composition_cache.lock() {
                Ok(mut cache) => {
                    cache.insert(key, value);
                }
                Err(_) => runtime.set_status(RUNTIME_INTERNAL_ERROR),
            }
        }
        value
    }

    fn validate_compute(
        &self,
        node: &GraphLanguage,
        egraph: &Graph,
        runtime: &RuntimeState,
    ) -> bool {
        if !runtime.is_running() || !is_compute_kind(node.kind) {
            return false;
        }
        let operand_types: Vec<u32> = node
            .children
            .iter()
            .map(|child| egraph[egraph.find(*child)].data.type_id)
            .collect();
        let key = ComputeValidationKey {
            semantic_id: node.semantic_id,
            kind: node.kind,
            result_type: node.type_id,
            relations: node.relations.to_vec(),
            operand_types: operand_types.clone(),
            data_input_count: node.data_input_count,
        };
        match runtime.compute_validation_cache.lock() {
            Ok(cache) => {
                if let Some(cached) = cache.get(&key) {
                    return *cached;
                }
            }
            Err(_) => {
                runtime.set_status(RUNTIME_INTERNAL_ERROR);
                return false;
            }
        }
        let status = unsafe {
            (self.validate_compute_fn)(
                self.context(),
                node.semantic_id,
                node.kind,
                node.type_id,
                node.relations.as_ptr(),
                operand_types.as_ptr(),
                node.relations.len() as u64,
                node.data_input_count,
            )
        };
        let valid = runtime.record_callback_status(status);
        if runtime.is_running() {
            match runtime.compute_validation_cache.lock() {
                Ok(mut cache) => {
                    cache.insert(key, valid);
                }
                Err(_) => runtime.set_status(RUNTIME_INTERNAL_ERROR),
            }
        }
        valid
    }

    fn validate_concat(
        &self,
        node: &GraphLanguage,
        egraph: &Graph,
        runtime: &RuntimeState,
    ) -> bool {
        if !runtime.is_running() || node.kind != NODE_CONCAT {
            return false;
        }
        let child_types: Vec<u32> = node
            .children
            .iter()
            .map(|child| egraph[egraph.find(*child)].data.type_id)
            .collect();
        let key = ConcatValidationKey {
            result_type: node.type_id,
            axis: node.axis,
            input_types: child_types.clone(),
        };
        match runtime.concat_validation_cache.lock() {
            Ok(cache) => {
                if let Some(cached) = cache.get(&key) {
                    return *cached;
                }
            }
            Err(_) => {
                runtime.set_status(RUNTIME_INTERNAL_ERROR);
                return false;
            }
        }
        let status = unsafe {
            (self.validate_concat_fn)(
                self.context(),
                node.type_id,
                node.axis,
                child_types.as_ptr(),
                child_types.len() as u64,
            )
        };
        let valid = runtime.record_callback_status(status);
        if runtime.is_running() {
            match runtime.concat_validation_cache.lock() {
                Ok(mut cache) => {
                    cache.insert(key, valid);
                }
                Err(_) => runtime.set_status(RUNTIME_INTERNAL_ERROR),
            }
        }
        valid
    }

    fn factor_concat(
        &self,
        result_type: u32,
        axis: i32,
        relations: &[u32],
        source_types: &[u32],
        runtime: &RuntimeState,
    ) -> Option<(u32, u32, i32)> {
        if !runtime.is_running() || relations.len() != source_types.len() {
            return None;
        }
        let key = ConcatFactorKey {
            result_type,
            axis,
            relations: relations.to_vec(),
            source_types: source_types.to_vec(),
        };
        match runtime.concat_factor_cache.lock() {
            Ok(cache) => {
                if let Some(cached) = cache.get(&key) {
                    return *cached;
                }
            }
            Err(_) => {
                runtime.set_status(RUNTIME_INTERNAL_ERROR);
                return None;
            }
        }
        let mut result_relation = 0;
        let mut source_concat_type = 0;
        let mut source_axis = -1;
        let status = unsafe {
            (self.factor_concat_fn)(
                self.context(),
                result_type,
                axis,
                relations.as_ptr(),
                source_types.as_ptr(),
                relations.len() as u64,
                &mut result_relation,
                &mut source_concat_type,
                &mut source_axis,
            )
        };
        let value = if runtime.record_callback_status(status)
            && result_relation != 0
            && source_concat_type != 0
            && source_axis >= 0
        {
            Some((result_relation, source_concat_type, source_axis))
        } else {
            if status == CALLBACK_EXACT {
                runtime.set_status(RUNTIME_INTERNAL_ERROR);
            }
            None
        };
        if runtime.is_running() {
            match runtime.concat_factor_cache.lock() {
                Ok(mut cache) => {
                    cache.insert(key, value);
                }
                Err(_) => runtime.set_status(RUNTIME_INTERNAL_ERROR),
            }
        }
        value
    }

    fn reindex_compute(
        &self,
        compute: &GraphLanguage,
        destination_type: u32,
        relation: u32,
        runtime: &RuntimeState,
    ) -> Option<(Vec<u32>, bool)> {
        if !runtime.is_running() || !is_compute_kind(compute.kind) {
            return None;
        }
        let key = ReindexKey {
            semantic_id: compute.semantic_id,
            kind: compute.kind,
            source_type: compute.type_id,
            destination_type,
            result_relation: relation,
            input_relations: compute.relations.to_vec(),
            data_input_count: compute.data_input_count,
        };
        match runtime.reindex_cache.lock() {
            Ok(cache) => {
                if let Some(cached) = cache.get(&key) {
                    return cached.clone();
                }
            }
            Err(_) => {
                runtime.set_status(RUNTIME_INTERNAL_ERROR);
                return None;
            }
        }
        let mut output = vec![0; compute.relations.len()];
        let mut reindex_read_init = 0;
        let status = unsafe {
            (self.reindex_compute_fn)(
                self.context(),
                compute.semantic_id,
                compute.kind,
                compute.type_id,
                destination_type,
                relation,
                compute.relations.as_ptr(),
                compute.relations.len() as u64,
                compute.data_input_count,
                output.as_mut_ptr(),
                &mut reindex_read_init,
            )
        };
        let value = if runtime.record_callback_status(status)
            && output.iter().all(|id| *id != 0)
            && reindex_read_init <= 1
        {
            Some((output, reindex_read_init != 0))
        } else {
            if status == CALLBACK_EXACT {
                runtime.set_status(RUNTIME_INTERNAL_ERROR);
            }
            None
        };
        if runtime.is_running() {
            match runtime.reindex_cache.lock() {
                Ok(mut cache) => {
                    cache.insert(key, value.clone());
                }
                Err(_) => runtime.set_status(RUNTIME_INTERNAL_ERROR),
            }
        }
        value
    }

    fn reparameterize_elementwise(
        &self,
        compute: &GraphLanguage,
        destination_type: u32,
        relation: u32,
        runtime: &RuntimeState,
    ) -> Option<(Vec<u32>, Vec<u32>, Vec<u32>)> {
        if !runtime.is_running() || compute.kind != NODE_ELEMENTWISE {
            return None;
        }
        let key = ReindexKey {
            semantic_id: compute.semantic_id,
            kind: compute.kind,
            source_type: compute.type_id,
            destination_type,
            result_relation: relation,
            input_relations: compute.relations.to_vec(),
            data_input_count: compute.data_input_count,
        };
        match runtime.reparameterization_cache.lock() {
            Ok(cache) => {
                if let Some(cached) = cache.get(&key) {
                    return cached.clone();
                }
            }
            Err(_) => {
                runtime.set_status(RUNTIME_INTERNAL_ERROR);
                return None;
            }
        }
        let mut compute_relations = vec![0; compute.relations.len()];
        let mut access_relations = vec![0; compute.relations.len()];
        let mut access_types = vec![0; compute.relations.len()];
        let status = unsafe {
            (self.reparameterize_elementwise_fn)(
                self.context(),
                compute.semantic_id,
                compute.type_id,
                destination_type,
                relation,
                compute.relations.as_ptr(),
                compute.relations.len() as u64,
                compute.data_input_count,
                compute_relations.as_mut_ptr(),
                access_relations.as_mut_ptr(),
                access_types.as_mut_ptr(),
            )
        };
        let value = if runtime.record_callback_status(status)
            && compute_relations.iter().all(|id| *id != 0)
            && access_relations
                .iter()
                .zip(&access_types)
                .all(|(relation, ty)| (*relation == 0) == (*ty == 0))
        {
            Some((compute_relations, access_relations, access_types))
        } else {
            if status == CALLBACK_EXACT {
                runtime.set_status(RUNTIME_INTERNAL_ERROR);
            }
            None
        };
        if runtime.is_running() {
            match runtime.reparameterization_cache.lock() {
                Ok(mut cache) => {
                    cache.insert(key, value.clone());
                }
                Err(_) => runtime.set_status(RUNTIME_INTERNAL_ERROR),
            }
        }
        value
    }

    fn reparameterize_reshape_compute(
        &self,
        compute: &GraphLanguage,
        access_operand: usize,
        access_relation: u32,
        runtime: &RuntimeState,
    ) -> Option<ReshapeComputeValue> {
        if !runtime.is_running()
            || access_operand >= compute.data_input_count as usize
            || access_operand >= compute.relations.len()
        {
            return None;
        }
        let key = ReshapeComputeKey {
            semantic_id: compute.semantic_id,
            kind: compute.kind,
            result_type: compute.type_id,
            access_operand: access_operand as u32,
            access_relation,
            input_relations: compute.relations.to_vec(),
            data_input_count: compute.data_input_count,
        };
        match runtime.reshape_compute_cache.lock() {
            Ok(cache) => {
                if let Some(cached) = cache.get(&key) {
                    return cached.clone();
                }
            }
            Err(_) => {
                runtime.set_status(RUNTIME_INTERNAL_ERROR);
                return None;
            }
        }
        let mut inner_result_type = 0;
        let mut outer_result_relation = 0;
        let mut compute_relations = vec![0; compute.relations.len()];
        let mut access_relations = vec![0; compute.relations.len()];
        let mut access_types = vec![0; compute.relations.len()];
        let status = unsafe {
            (self.reparameterize_reshape_compute_fn)(
                self.context(),
                compute.semantic_id,
                compute.kind,
                compute.type_id,
                access_operand as u32,
                access_relation,
                compute.relations.as_ptr(),
                compute.relations.len() as u64,
                compute.data_input_count,
                &mut inner_result_type,
                &mut outer_result_relation,
                compute_relations.as_mut_ptr(),
                access_relations.as_mut_ptr(),
                access_types.as_mut_ptr(),
            )
        };
        let value = if runtime.record_callback_status(status)
            && inner_result_type != 0
            && outer_result_relation != 0
            && compute_relations.iter().all(|id| *id != 0)
            && access_relations
                .iter()
                .zip(&access_types)
                .all(|(relation, ty)| (*relation == 0) == (*ty == 0))
        {
            Some((
                inner_result_type,
                outer_result_relation,
                compute_relations,
                access_relations,
                access_types,
            ))
        } else {
            if status == CALLBACK_EXACT {
                runtime.set_status(RUNTIME_INTERNAL_ERROR);
            }
            None
        };
        if runtime.is_running() {
            match runtime.reshape_compute_cache.lock() {
                Ok(mut cache) => {
                    cache.insert(key, value.clone());
                }
                Err(_) => runtime.set_status(RUNTIME_INTERNAL_ERROR),
            }
        }
        value
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ClassFacts {
    type_id: u32,
    valid: bool,
}

struct GraphAnalysis;

type Graph = EGraph<GraphLanguage, GraphAnalysis>;

impl Analysis<GraphLanguage> for GraphAnalysis {
    type Data = ClassFacts;

    fn make(_egraph: &mut Graph, node: &GraphLanguage, _id: Id) -> Self::Data {
        ClassFacts {
            type_id: node.type_id,
            valid: valid_node_shape(node),
        }
    }

    fn merge(&mut self, current: &mut Self::Data, other: Self::Data) -> DidMerge {
        let before = current.clone();
        current.valid &= other.valid && current.type_id == other.type_id;
        DidMerge(*current != before, *current != other)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct GraphCost {
    invalid: u64,
    compute_occurrences: u64,
    compute_ids: Vec<u32>,
    access_occurrences: u64,
    concat_occurrences: u64,
    nodes: u64,
    canonical_key: Vec<u32>,
}

impl Ord for GraphCost {
    fn cmp(&self, other: &Self) -> Ordering {
        (
            self.invalid,
            self.compute_occurrences,
            &self.compute_ids,
            self.access_occurrences,
            self.concat_occurrences,
            self.nodes,
            &self.canonical_key,
        )
            .cmp(&(
                other.invalid,
                other.compute_occurrences,
                &other.compute_ids,
                other.access_occurrences,
                other.concat_occurrences,
                other.nodes,
                &other.canonical_key,
            ))
    }
}

impl PartialOrd for GraphCost {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

struct GraphCostFunction {
    service: RelationService,
    runtime: Arc<RuntimeState>,
    calls: Rc<Cell<u64>>,
}

impl CostFunction<GraphLanguage> for GraphCostFunction {
    type Cost = GraphCost;

    fn cost<C>(&mut self, node: &GraphLanguage, mut child_cost: C) -> Self::Cost
    where
        C: FnMut(Id) -> Self::Cost,
    {
        self.calls.set(self.calls.get().saturating_add(1));
        let mut result = GraphCost {
            invalid: 0,
            compute_occurrences: u64::from(is_compute_kind(node.kind)),
            compute_ids: if is_compute_kind(node.kind) {
                vec![node.semantic_id]
            } else {
                Vec::new()
            },
            access_occurrences: u64::from(node.kind == NODE_ACCESS),
            concat_occurrences: u64::from(node.kind == NODE_CONCAT),
            nodes: 1,
            canonical_key: vec![
                node.kind,
                node.semantic_id,
                node.type_id,
                node.data_input_count,
                (i64::from(node.axis) + 1) as u32,
                node.source_order,
                node.children.len() as u32,
                node.relations.len() as u32,
            ],
        };
        result.canonical_key.extend(node.relations.iter().copied());
        if node.kind == NODE_ACCESS {
            match self.service.facts(node.relations[0], &self.runtime) {
                Some(facts) if facts.flags & RELATION_MATERIALIZABLE != 0 => {}
                _ => result.invalid = 1,
            }
        }
        for child in node.children.iter().copied() {
            let cost = child_cost(child);
            result.invalid = result.invalid.saturating_add(cost.invalid);
            result.compute_occurrences = result
                .compute_occurrences
                .saturating_add(cost.compute_occurrences);
            result.compute_ids.extend(cost.compute_ids);
            result.access_occurrences = result
                .access_occurrences
                .saturating_add(cost.access_occurrences);
            result.concat_occurrences = result
                .concat_occurrences
                .saturating_add(cost.concat_occurrences);
            result.nodes = result.nodes.saturating_add(cost.nodes);
            result.canonical_key.extend(cost.canonical_key);
        }
        result.compute_ids.sort_unstable();
        result
    }
}

#[derive(Clone, Copy, Debug)]
enum RuleKind {
    Identity,
    Composition,
    ComputeAbsorption,
    Concat,
    ResultReindex,
    ReshapeThroughCompute,
}

impl RuleKind {
    fn name(self) -> &'static str {
        match self {
            Self::Identity => "wafer-access-identity",
            Self::Composition => "wafer-access-composition",
            Self::ComputeAbsorption => "wafer-compute-operand-access",
            Self::Concat => "wafer-concat-normalization",
            Self::ResultReindex => "wafer-compute-result-reindex",
            Self::ReshapeThroughCompute => "wafer-reshape-through-compute",
        }
    }
}

struct DynamicSearcher {
    kind: RuleKind,
    service: RelationService,
    runtime: Arc<RuntimeState>,
}

impl DynamicSearcher {
    fn may_match(&self, egraph: &Graph, eclass: Id) -> bool {
        for node in &egraph[eclass].nodes {
            match self.kind {
                RuleKind::Identity if node.kind == NODE_ACCESS => return true,
                RuleKind::Composition if node.kind == NODE_ACCESS => {
                    let child = egraph.find(node.children[0]);
                    if egraph[child]
                        .nodes
                        .iter()
                        .any(|node| node.kind == NODE_ACCESS)
                    {
                        return true;
                    }
                }
                RuleKind::ComputeAbsorption if is_compute_kind(node.kind) => {
                    for child in node.children.iter().take(node.data_input_count as usize) {
                        let child = egraph.find(*child);
                        if egraph[child]
                            .nodes
                            .iter()
                            .any(|node| node.kind == NODE_ACCESS)
                        {
                            return true;
                        }
                    }
                }
                RuleKind::Concat if node.kind == NODE_CONCAT => return true,
                RuleKind::ResultReindex if node.kind == NODE_ACCESS => {
                    let child = egraph.find(node.children[0]);
                    if egraph[child]
                        .nodes
                        .iter()
                        .any(|node| is_compute_kind(node.kind))
                    {
                        return true;
                    }
                }
                RuleKind::ReshapeThroughCompute if is_compute_kind(node.kind) => {
                    for child in node.children.iter().take(node.data_input_count as usize) {
                        let child = egraph.find(*child);
                        for access in egraph[child]
                            .nodes
                            .iter()
                            .filter(|node| node.kind == NODE_ACCESS)
                        {
                            if self
                                .service
                                .facts(access.relations[0], &self.runtime)
                                .is_some_and(|facts| facts.flags & RELATION_CANONICAL_RESHAPE != 0)
                            {
                                return true;
                            }
                        }
                    }
                }
                _ => {}
            }
        }
        false
    }
}

impl Searcher<GraphLanguage, GraphAnalysis> for DynamicSearcher {
    fn search_eclass_with_limit(
        &self,
        egraph: &Graph,
        eclass: Id,
        limit: usize,
    ) -> Option<SearchMatches<'_, GraphLanguage>> {
        if limit == 0 || !self.may_match(egraph, eclass) {
            return None;
        }
        Some(SearchMatches {
            eclass,
            substs: vec![Subst::with_capacity(0)],
            ast: None,
        })
    }

    fn vars(&self) -> Vec<Var> {
        Vec::new()
    }
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct AccessChoice {
    relation: u32,
    child: Id,
    child_type: u32,
}

fn enumerate_access_combinations(
    options: &[Vec<AccessChoice>],
    index: usize,
    current: &mut Vec<AccessChoice>,
    output: &mut Vec<Vec<AccessChoice>>,
) -> bool {
    if output.len() > MAX_CONCAT_ACCESS_COMBINATIONS {
        return false;
    }
    if index == options.len() {
        output.push(current.clone());
        return output.len() <= MAX_CONCAT_ACCESS_COMBINATIONS;
    }
    for choice in &options[index] {
        current.push(choice.clone());
        if !enumerate_access_combinations(options, index + 1, current, output) {
            return false;
        }
        current.pop();
    }
    true
}

struct DynamicApplier {
    kind: RuleKind,
    service: RelationService,
    runtime: Arc<RuntimeState>,
    previous_inputs: Mutex<BTreeMap<Id, ApplicationReadSet>>,
}

// An applier reads the matched nodes, their child nodes, and the type facts
// at the resulting frontier. Capture actual records instead of using an
// epoch/node-count heuristic: a union can expose new alternatives without
// changing either of those quantities.
#[derive(Eq, PartialEq)]
struct ClassReadSet {
    id: Id,
    facts: ClassFacts,
    nodes: Vec<GraphLanguage>,
}

impl ClassReadSet {
    fn capture(egraph: &Graph, id: Id) -> Self {
        let id = egraph.find(id);
        let mut nodes = egraph[id].nodes.clone();
        for node in &mut nodes {
            for child in node.children.iter_mut() {
                *child = egraph.find(*child);
            }
        }
        Self {
            id,
            facts: egraph[id].data.clone(),
            nodes,
        }
    }
}

#[derive(Eq, PartialEq)]
struct ApplicationReadSet {
    root: ClassReadSet,
    children: Vec<ClassReadSet>,
    frontier: Vec<(Id, ClassFacts)>,
}

impl ApplicationReadSet {
    fn capture(egraph: &Graph, id: Id) -> Self {
        let root = ClassReadSet::capture(egraph, id);
        let child_ids: BTreeSet<Id> = root
            .nodes
            .iter()
            .flat_map(|node| node.children.iter().copied())
            .collect();
        let children: Vec<_> = child_ids
            .into_iter()
            .map(|child| ClassReadSet::capture(egraph, child))
            .collect();
        let frontier_ids: BTreeSet<Id> = children
            .iter()
            .flat_map(|child| child.nodes.iter())
            .flat_map(|node| node.children.iter().copied())
            .collect();
        let frontier = frontier_ids
            .into_iter()
            .map(|child| (child, egraph[child].data.clone()))
            .collect();
        Self {
            root,
            children,
            frontier,
        }
    }
}

impl DynamicApplier {
    fn merge_target(&self, egraph: &mut Graph, eclass: Id, target: Id) -> bool {
        if egraph.union(eclass, target) {
            match self.kind {
                RuleKind::Identity => RuntimeState::increment(&self.runtime.identity_applications),
                RuleKind::Composition => {
                    RuntimeState::increment(&self.runtime.composition_applications)
                }
                RuleKind::ComputeAbsorption => {
                    RuntimeState::increment(&self.runtime.compute_absorption_applications)
                }
                RuleKind::Concat => RuntimeState::increment(&self.runtime.concat_applications),
                RuleKind::ResultReindex => {
                    RuntimeState::increment(&self.runtime.result_reindex_applications)
                }
                RuleKind::ReshapeThroughCompute => {
                    RuntimeState::increment(&self.runtime.reshape_through_compute_applications)
                }
            }
            true
        } else {
            false
        }
    }

    fn apply_identity(&self, egraph: &mut Graph, eclass: Id) -> Vec<Id> {
        let nodes = egraph[eclass].nodes.clone();
        let mut applied = Vec::new();
        for node in nodes.into_iter().filter(|node| node.kind == NODE_ACCESS) {
            if let Some(facts) = self.service.facts(node.relations[0], &self.runtime) {
                if facts.flags & RELATION_IDENTITY != 0 {
                    let target = egraph.find(node.children[0]);
                    if self.merge_target(egraph, eclass, target) {
                        applied.push(target);
                    }
                }
            }
        }
        applied
    }

    fn apply_composition(&self, egraph: &mut Graph, eclass: Id) -> Vec<Id> {
        let nodes = egraph[eclass].nodes.clone();
        let mut applied = Vec::new();
        for outer in nodes.into_iter().filter(|node| node.kind == NODE_ACCESS) {
            let child_class = egraph.find(outer.children[0]);
            let inner_nodes = egraph[child_class].nodes.clone();
            for inner in inner_nodes
                .into_iter()
                .filter(|node| node.kind == NODE_ACCESS)
            {
                let Some(relation) =
                    self.service
                        .compose(outer.relations[0], inner.relations[0], &self.runtime)
                else {
                    continue;
                };
                let target = egraph.add(GraphLanguage {
                    kind: NODE_ACCESS,
                    semantic_id: 0,
                    type_id: outer.type_id,
                    relations: vec![relation].into_boxed_slice(),
                    data_input_count: 0,
                    axis: -1,
                    source_order: 0,
                    children: vec![egraph.find(inner.children[0])].into_boxed_slice(),
                });
                if self.merge_target(egraph, eclass, target) {
                    applied.push(target);
                }
            }
        }
        applied
    }

    fn apply_compute_absorption(&self, egraph: &mut Graph, eclass: Id) -> Vec<Id> {
        let nodes = egraph[eclass].nodes.clone();
        let mut applied = Vec::new();
        for compute in nodes.into_iter().filter(|node| is_compute_kind(node.kind)) {
            for operand in 0..compute.data_input_count as usize {
                let child_class = egraph.find(compute.children[operand]);
                let access_nodes = egraph[child_class].nodes.clone();
                for access in access_nodes
                    .into_iter()
                    .filter(|node| node.kind == NODE_ACCESS)
                {
                    let Some(relation) = self.service.compose(
                        compute.relations[operand],
                        access.relations[0],
                        &self.runtime,
                    ) else {
                        continue;
                    };
                    let mut target = compute.clone();
                    target.relations[operand] = relation;
                    target.children[operand] = egraph.find(access.children[0]);
                    if !self
                        .service
                        .validate_compute(&target, egraph, &self.runtime)
                    {
                        continue;
                    }
                    let target = egraph.add(target);
                    if self.merge_target(egraph, eclass, target) {
                        applied.push(target);
                    }
                }
            }
        }
        applied
    }

    fn apply_concat(&self, egraph: &mut Graph, eclass: Id) -> Vec<Id> {
        let nodes = egraph[eclass].nodes.clone();
        let mut applied = Vec::new();
        for concat in nodes.into_iter().filter(|node| node.kind == NODE_CONCAT) {
            if concat.children.len() == 1 {
                let target = egraph.find(concat.children[0]);
                if egraph[target].data.type_id == concat.type_id
                    && self.merge_target(egraph, eclass, target)
                {
                    applied.push(target);
                }
            }

            for index in 0..concat.children.len() {
                let child_class = egraph.find(concat.children[index]);
                let inner_nodes = egraph[child_class].nodes.clone();
                for inner in inner_nodes
                    .into_iter()
                    .filter(|node| node.kind == NODE_CONCAT && node.axis == concat.axis)
                {
                    let mut target = concat.clone();
                    target.children = concat.children[..index]
                        .iter()
                        .copied()
                        .chain(inner.children.iter().copied())
                        .chain(concat.children[index + 1..].iter().copied())
                        .map(|child| egraph.find(child))
                        .collect::<Vec<_>>()
                        .into_boxed_slice();
                    if !self.service.validate_concat(&target, egraph, &self.runtime) {
                        continue;
                    }
                    let target = egraph.add(target);
                    if self.merge_target(egraph, eclass, target) {
                        applied.push(target);
                    }
                }
            }

            if concat.children.len() < 2 {
                continue;
            }
            let mut options = Vec::with_capacity(concat.children.len());
            for child in &concat.children {
                let child_class = egraph.find(*child);
                let mut choices = Vec::new();
                for access in egraph[child_class]
                    .nodes
                    .iter()
                    .filter(|node| node.kind == NODE_ACCESS)
                {
                    let base = egraph.find(access.children[0]);
                    choices.push(AccessChoice {
                        relation: access.relations[0],
                        child: base,
                        child_type: egraph[base].data.type_id,
                    });
                }
                choices.sort();
                choices.dedup();
                if choices.is_empty() {
                    options.clear();
                    break;
                }
                options.push(choices);
            }
            if options.is_empty() {
                continue;
            }
            let mut combinations = Vec::new();
            if !enumerate_access_combinations(&options, 0, &mut Vec::new(), &mut combinations) {
                self.runtime.set_status(RUNTIME_BUDGET);
                return applied;
            }
            for combination in combinations {
                let relations: Vec<u32> = combination.iter().map(|item| item.relation).collect();
                let source_types: Vec<u32> =
                    combination.iter().map(|item| item.child_type).collect();
                let Some((relation, source_type, source_axis)) = self.service.factor_concat(
                    concat.type_id,
                    concat.axis,
                    &relations,
                    &source_types,
                    &self.runtime,
                ) else {
                    continue;
                };
                let source_concat = GraphLanguage {
                    kind: NODE_CONCAT,
                    semantic_id: 0,
                    type_id: source_type,
                    relations: Box::new([]),
                    data_input_count: 0,
                    axis: source_axis,
                    source_order: 0,
                    children: combination
                        .iter()
                        .map(|item| egraph.find(item.child))
                        .collect::<Vec<_>>()
                        .into_boxed_slice(),
                };
                if !self
                    .service
                    .validate_concat(&source_concat, egraph, &self.runtime)
                {
                    continue;
                }
                let source_concat = egraph.add(source_concat);
                let target = egraph.add(GraphLanguage {
                    kind: NODE_ACCESS,
                    semantic_id: 0,
                    type_id: concat.type_id,
                    relations: vec![relation].into_boxed_slice(),
                    data_input_count: 0,
                    axis: -1,
                    source_order: 0,
                    children: vec![source_concat].into_boxed_slice(),
                });
                if self.merge_target(egraph, eclass, target) {
                    applied.push(target);
                }
            }
        }
        applied
    }

    fn apply_result_reindex(&self, egraph: &mut Graph, eclass: Id) -> Vec<Id> {
        let nodes = egraph[eclass].nodes.clone();
        let mut applied = Vec::new();
        for access in nodes.into_iter().filter(|node| node.kind == NODE_ACCESS) {
            let child_class = egraph.find(access.children[0]);
            let computes = egraph[child_class].nodes.clone();
            for compute in computes
                .into_iter()
                .filter(|node| is_compute_kind(node.kind))
            {
                if let Some((relations, reindex_read_init)) = self.service.reindex_compute(
                    &compute,
                    access.type_id,
                    access.relations[0],
                    &self.runtime,
                ) {
                    let mut target = compute.clone();
                    target.type_id = access.type_id;
                    target.relations = relations.into_boxed_slice();
                    if reindex_read_init {
                        for init in compute.data_input_count as usize..target.children.len() {
                            let reindexed_init = egraph.add(GraphLanguage {
                                kind: NODE_ACCESS,
                                semantic_id: 0,
                                type_id: access.type_id,
                                relations: access.relations.clone(),
                                data_input_count: 0,
                                axis: -1,
                                source_order: 0,
                                children: vec![egraph.find(target.children[init])]
                                    .into_boxed_slice(),
                            });
                            target.children[init] = reindexed_init;
                        }
                    }
                    if self
                        .service
                        .validate_compute(&target, egraph, &self.runtime)
                    {
                        let target = egraph.add(target);
                        if self.merge_target(egraph, eclass, target) {
                            applied.push(target);
                        }
                    }
                }

                if let Some((relations, access_relations, access_types)) =
                    self.service.reparameterize_elementwise(
                        &compute,
                        access.type_id,
                        access.relations[0],
                        &self.runtime,
                    )
                {
                    let mut target = compute.clone();
                    target.type_id = access.type_id;
                    target.relations = relations.into_boxed_slice();
                    for operand in 0..target.children.len() {
                        if access_relations[operand] == 0 {
                            continue;
                        }
                        let reindexed_operand = egraph.add(GraphLanguage {
                            kind: NODE_ACCESS,
                            semantic_id: 0,
                            type_id: access_types[operand],
                            relations: vec![access_relations[operand]].into_boxed_slice(),
                            data_input_count: 0,
                            axis: -1,
                            source_order: 0,
                            children: vec![egraph.find(target.children[operand])]
                                .into_boxed_slice(),
                        });
                        target.children[operand] = reindexed_operand;
                    }
                    if self
                        .service
                        .validate_compute(&target, egraph, &self.runtime)
                    {
                        let target = egraph.add(target);
                        if self.merge_target(egraph, eclass, target) {
                            applied.push(target);
                        }
                    }
                }
            }
        }
        applied
    }

    fn apply_reshape_through_compute(&self, egraph: &mut Graph, eclass: Id) -> Vec<Id> {
        let nodes = egraph[eclass].nodes.clone();
        let mut applied = Vec::new();
        for compute in nodes.into_iter().filter(|node| is_compute_kind(node.kind)) {
            for operand in 0..compute.data_input_count as usize {
                let child_class = egraph.find(compute.children[operand]);
                let accesses = egraph[child_class].nodes.clone();
                for access in accesses.into_iter().filter(|node| node.kind == NODE_ACCESS) {
                    let Some((
                        inner_result_type,
                        outer_result_relation,
                        compute_relations,
                        operand_access_relations,
                        operand_access_types,
                    )) = self.service.reparameterize_reshape_compute(
                        &compute,
                        operand,
                        access.relations[0],
                        &self.runtime,
                    )
                    else {
                        continue;
                    };
                    let mut inner = compute.clone();
                    inner.type_id = inner_result_type;
                    inner.relations = compute_relations.into_boxed_slice();
                    inner.children[operand] = egraph.find(access.children[0]);
                    for index in 0..inner.children.len() {
                        if index == operand || operand_access_relations[index] == 0 {
                            continue;
                        }
                        let value = egraph.add(GraphLanguage {
                            kind: NODE_ACCESS,
                            semantic_id: 0,
                            type_id: operand_access_types[index],
                            relations: vec![operand_access_relations[index]].into_boxed_slice(),
                            data_input_count: 0,
                            axis: -1,
                            source_order: 0,
                            children: vec![egraph.find(inner.children[index])].into_boxed_slice(),
                        });
                        inner.children[index] = value;
                    }
                    if !self.service.validate_compute(&inner, egraph, &self.runtime) {
                        continue;
                    }
                    let inner = egraph.add(inner);
                    let target = egraph.add(GraphLanguage {
                        kind: NODE_ACCESS,
                        semantic_id: 0,
                        type_id: compute.type_id,
                        relations: vec![outer_result_relation].into_boxed_slice(),
                        data_input_count: 0,
                        axis: -1,
                        source_order: compute.source_order,
                        children: vec![inner].into_boxed_slice(),
                    });
                    if self.merge_target(egraph, eclass, target) {
                        applied.push(target);
                    }
                }
            }
        }
        applied
    }
}

impl Applier<GraphLanguage, GraphAnalysis> for DynamicApplier {
    fn apply_one(
        &self,
        egraph: &mut Graph,
        eclass: Id,
        _substitution: &Subst,
        _searcher_ast: Option<&PatternAst<GraphLanguage>>,
        _rule_name: Symbol,
    ) -> Vec<Id> {
        if !self.runtime.is_running() {
            return Vec::new();
        }
        RuntimeState::increment(&self.runtime.application_checks);
        let read_set = ApplicationReadSet::capture(egraph, eclass);
        match self.previous_inputs.lock() {
            Ok(mut previous) => {
                if previous.get(&read_set.root.id) == Some(&read_set) {
                    RuntimeState::increment(&self.runtime.unchanged_applications);
                    return Vec::new();
                }
                // Store the pre-application state. Newly created alternatives
                // must be visible to this same rule on its next invocation.
                previous.insert(read_set.root.id, read_set);
            }
            Err(_) => {
                self.runtime.set_status(RUNTIME_INTERNAL_ERROR);
                return Vec::new();
            }
        }
        match self.kind {
            RuleKind::Identity => self.apply_identity(egraph, eclass),
            RuleKind::Composition => self.apply_composition(egraph, eclass),
            RuleKind::ComputeAbsorption => self.apply_compute_absorption(egraph, eclass),
            RuleKind::Concat => self.apply_concat(egraph, eclass),
            RuleKind::ResultReindex => self.apply_result_reindex(egraph, eclass),
            RuleKind::ReshapeThroughCompute => self.apply_reshape_through_compute(egraph, eclass),
        }
    }

    fn vars(&self) -> Vec<Var> {
        Vec::new()
    }
}

#[derive(Default)]
struct SchedulerCounters {
    matches: Cell<u64>,
    merges: Cell<u64>,
    exhausted: Cell<bool>,
}

struct WorkBudgetScheduler {
    maximum_matches: u64,
    counters: Rc<SchedulerCounters>,
}

impl RewriteScheduler<GraphLanguage, GraphAnalysis> for WorkBudgetScheduler {
    fn search_rewrite<'a>(
        &mut self,
        _iteration: usize,
        egraph: &Graph,
        rewrite: &'a Rewrite<GraphLanguage, GraphAnalysis>,
    ) -> Vec<SearchMatches<'a, GraphLanguage>> {
        let used = self.counters.matches.get();
        if used >= self.maximum_matches {
            self.counters.exhausted.set(true);
            return Vec::new();
        }
        let remaining = self.maximum_matches - used;
        let probe_limit = usize::try_from(remaining.saturating_add(1)).unwrap_or(usize::MAX);
        let mut matches = rewrite.search_with_limit(egraph, probe_limit);
        let count = matches
            .iter()
            .map(|candidate| candidate.substs.len() as u64)
            .sum::<u64>();
        if count > remaining {
            self.counters.exhausted.set(true);
            let mut keep = usize::try_from(remaining).unwrap_or(usize::MAX);
            matches.retain_mut(|candidate| {
                if keep == 0 {
                    return false;
                }
                if candidate.substs.len() > keep {
                    candidate.substs.truncate(keep);
                }
                keep -= candidate.substs.len();
                true
            });
        }
        self.counters
            .matches
            .set(used.saturating_add(count.min(remaining)));
        matches
    }

    fn apply_rewrite(
        &mut self,
        _iteration: usize,
        egraph: &mut Graph,
        rewrite: &Rewrite<GraphLanguage, GraphAnalysis>,
        matches: Vec<SearchMatches<GraphLanguage>>,
    ) -> usize {
        let applied = rewrite.apply(egraph, &matches).len();
        self.counters
            .merges
            .set(self.counters.merges.get().saturating_add(applied as u64));
        applied
    }
}

fn checked_count(value: u64) -> Result<usize, u32> {
    usize::try_from(value).map_err(|_| STATUS_INVALID_INPUT)
}

unsafe fn required_slice<'a, T>(pointer: *const T, count: usize) -> Result<&'a [T], u32> {
    if count == 0 {
        return Ok(&[]);
    }
    if pointer.is_null() {
        return Err(STATUS_INVALID_INPUT);
    }
    Ok(slice::from_raw_parts(pointer, count))
}

fn raw_graph_cost(
    roots: &[u32],
    nodes: &[WaferEGraphNode],
    children: &[u32],
    relations: &[u32],
    service: RelationService,
    runtime: &RuntimeState,
) -> Result<GraphCost, u32> {
    if roots.is_empty() {
        return Err(STATUS_INVALID_INPUT);
    }
    let mut reachable = vec![false; nodes.len()];
    let mut worklist: Vec<usize> = roots.iter().map(|root| *root as usize).collect();
    while let Some(index) = worklist.pop() {
        if index >= nodes.len() {
            return Err(STATUS_INVALID_INPUT);
        }
        if std::mem::replace(&mut reachable[index], true) {
            continue;
        }
        let node = nodes[index];
        let child_begin = node.child_offset as usize;
        let child_end = child_begin
            .checked_add(node.child_count as usize)
            .ok_or(STATUS_INVALID_INPUT)?;
        if child_end > children.len() {
            return Err(STATUS_INVALID_INPUT);
        }
        for child in &children[child_begin..child_end] {
            if *child as usize >= index {
                return Err(STATUS_INVALID_INPUT);
            }
            worklist.push(*child as usize);
        }
    }

    let mut cost = GraphCost {
        invalid: 0,
        compute_occurrences: 0,
        compute_ids: Vec::new(),
        access_occurrences: 0,
        concat_occurrences: 0,
        nodes: 0,
        canonical_key: roots.to_vec(),
    };
    for (index, node) in nodes.iter().copied().enumerate() {
        if !reachable[index] {
            continue;
        }
        let child_begin = node.child_offset as usize;
        let child_end = child_begin
            .checked_add(node.child_count as usize)
            .ok_or(STATUS_INVALID_INPUT)?;
        let relation_begin = node.relation_offset as usize;
        let relation_end = relation_begin
            .checked_add(node.relation_count as usize)
            .ok_or(STATUS_INVALID_INPUT)?;
        if child_end > children.len() || relation_end > relations.len() {
            return Err(STATUS_INVALID_INPUT);
        }
        let node_relations = &relations[relation_begin..relation_end];
        cost.nodes = cost.nodes.saturating_add(1);
        cost.compute_occurrences = cost
            .compute_occurrences
            .saturating_add(u64::from(is_compute_kind(node.kind)));
        if is_compute_kind(node.kind) {
            cost.compute_ids.push(node.semantic_id);
        }
        cost.access_occurrences = cost
            .access_occurrences
            .saturating_add(u64::from(node.kind == NODE_ACCESS));
        cost.concat_occurrences = cost
            .concat_occurrences
            .saturating_add(u64::from(node.kind == NODE_CONCAT));
        cost.canonical_key.extend([
            node.kind,
            node.semantic_id,
            node.type_id,
            node.data_input_count,
            (i64::from(node.axis) + 1) as u32,
            node.source_order,
            node.child_count,
            node.relation_count,
        ]);
        cost.canonical_key
            .extend(children[child_begin..child_end].iter().copied());
        cost.canonical_key.extend(node_relations.iter().copied());
        if node.kind == NODE_ACCESS {
            match service.facts(node_relations[0], runtime) {
                Some(facts) if facts.flags & RELATION_MATERIALIZABLE != 0 => {}
                _ => cost.invalid = cost.invalid.saturating_add(1),
            }
        }
    }
    cost.compute_ids.sort_unstable();
    Ok(cost)
}

fn is_strictly_better(candidate: &GraphCost, original: &GraphCost) -> bool {
    candidate.invalid == 0
        && candidate.compute_occurrences == original.compute_occurrences
        && candidate.compute_ids == original.compute_ids
        && (
            candidate.access_occurrences,
            candidate.concat_occurrences,
            candidate.nodes,
        ) < (
            original.access_occurrences,
            original.concat_occurrences,
            original.nodes,
        )
}

fn budget_stop(reason: &StopReason) -> bool {
    matches!(
        reason,
        StopReason::IterationLimit(_) | StopReason::NodeLimit(_) | StopReason::TimeLimit(_)
    )
}

unsafe fn run(request: &WaferEGraphRequest) -> Result<WaferEGraphResult, u32> {
    if request.schema_version != SCHEMA_VERSION || request.flags & !FLAG_FORCE_PANIC != 0 {
        return Err(STATUS_INVALID_INPUT);
    }
    if request.flags & FLAG_FORCE_PANIC != 0 {
        panic!("forced Wafer e-graph panic seam");
    }
    let service =
        RelationService::from_raw(request.relation_service).ok_or(STATUS_INVALID_INPUT)?;

    let node_count = checked_count(request.node_count)?;
    let child_count = checked_count(request.child_count)?;
    let relation_count = checked_count(request.relation_count)?;
    let root_count = checked_count(request.root_count)?;
    let nodes = required_slice(request.nodes, node_count)?;
    let children = required_slice(request.children, child_count)?;
    let relations = required_slice(request.relations, relation_count)?;
    let roots = required_slice(request.root_nodes, root_count)?;
    if node_count == 0
        || roots.is_empty()
        || roots.iter().any(|root| *root as usize >= node_count)
        || request.maximum_iterations == 0
        || request.maximum_e_nodes == 0
        || request.maximum_matches == 0
    {
        return Err(STATUS_INVALID_INPUT);
    }

    let runtime = Arc::new(RuntimeState::default());
    let original_cost = raw_graph_cost(roots, nodes, children, relations, service, &runtime)?;
    if runtime.status() == RUNTIME_BUDGET {
        return Ok(WaferEGraphResult::status(STATUS_BUDGET_EXHAUSTED));
    }
    if !runtime.is_running() {
        return Err(STATUS_INTERNAL_ERROR);
    }

    let mut egraph = Graph::new(GraphAnalysis);
    let mut ids = Vec::with_capacity(node_count);
    for (index, raw) in nodes.iter().copied().enumerate() {
        let child_begin = raw.child_offset as usize;
        let child_end = child_begin
            .checked_add(raw.child_count as usize)
            .ok_or(STATUS_INVALID_INPUT)?;
        let relation_begin = raw.relation_offset as usize;
        let relation_end = relation_begin
            .checked_add(raw.relation_count as usize)
            .ok_or(STATUS_INVALID_INPUT)?;
        if child_end > children.len() || relation_end > relations.len() {
            return Err(STATUS_INVALID_INPUT);
        }
        let mut child_ids = Vec::with_capacity(raw.child_count as usize);
        for child in &children[child_begin..child_end] {
            let child = *child as usize;
            if child >= index {
                return Err(STATUS_INVALID_INPUT);
            }
            child_ids.push(ids[child]);
        }
        let node = GraphLanguage {
            kind: raw.kind,
            semantic_id: raw.semantic_id,
            type_id: raw.type_id,
            relations: relations[relation_begin..relation_end]
                .to_vec()
                .into_boxed_slice(),
            data_input_count: raw.data_input_count,
            axis: raw.axis,
            source_order: raw.source_order,
            children: child_ids.into_boxed_slice(),
        };
        if !valid_node_shape(&node) || node.type_id == 0 {
            return Err(STATUS_INVALID_INPUT);
        }
        match node.kind {
            NODE_ACCESS => {
                let facts = service
                    .facts(node.relations[0], &runtime)
                    .ok_or(STATUS_INVALID_INPUT)?;
                let child_type = nodes[children[child_begin] as usize].type_id;
                if facts.destination_type_id != node.type_id || facts.source_type_id != child_type {
                    return Err(STATUS_INVALID_INPUT);
                }
            }
            kind if is_compute_kind(kind) => {
                if !service.validate_compute(&node, &egraph, &runtime) {
                    return if runtime.status() == RUNTIME_BUDGET {
                        Ok(WaferEGraphResult::status(STATUS_BUDGET_EXHAUSTED))
                    } else {
                        Err(STATUS_INVALID_INPUT)
                    };
                }
            }
            NODE_CONCAT => {
                let child_types: Vec<u32> = children[child_begin..child_end]
                    .iter()
                    .map(|child| nodes[*child as usize].type_id)
                    .collect();
                let status = (service.validate_concat_fn)(
                    service.context(),
                    node.type_id,
                    node.axis,
                    child_types.as_ptr(),
                    child_types.len() as u64,
                );
                if !runtime.record_callback_status(status) {
                    return if runtime.status() == RUNTIME_BUDGET {
                        Ok(WaferEGraphResult::status(STATUS_BUDGET_EXHAUSTED))
                    } else {
                        Err(STATUS_INVALID_INPUT)
                    };
                }
            }
            _ => {}
        }
        ids.push(egraph.add(node));
    }

    let rule_kinds = [
        RuleKind::Identity,
        RuleKind::Composition,
        RuleKind::ComputeAbsorption,
        RuleKind::Concat,
        RuleKind::ResultReindex,
        RuleKind::ReshapeThroughCompute,
    ];
    let rules: Vec<Rewrite<GraphLanguage, GraphAnalysis>> = rule_kinds
        .iter()
        .copied()
        .map(|kind| {
            Rewrite::new(
                kind.name(),
                DynamicSearcher {
                    kind,
                    service,
                    runtime: runtime.clone(),
                },
                DynamicApplier {
                    kind,
                    service,
                    runtime: runtime.clone(),
                    previous_inputs: Mutex::default(),
                },
            )
            .expect("fixed Wafer rewrite must be well formed")
        })
        .collect();
    let counters = Rc::new(SchedulerCounters::default());
    let scheduler = WorkBudgetScheduler {
        maximum_matches: request.maximum_matches,
        counters: counters.clone(),
    };
    let hook_runtime = runtime.clone();
    let node_limit = usize::try_from(request.maximum_e_nodes).unwrap_or(usize::MAX);
    let mut runner = Runner::<GraphLanguage, GraphAnalysis, ()>::new(GraphAnalysis)
        .with_egraph(egraph)
        .with_iter_limit(request.maximum_iterations as usize)
        .with_node_limit(node_limit)
        .with_time_limit(Duration::MAX)
        .with_scheduler(scheduler)
        .with_hook(move |_| {
            if hook_runtime.is_running() {
                Ok(())
            } else {
                Err("wafer relation service stopped equality saturation".to_string())
            }
        });
    runner
        .roots
        .extend(roots.iter().map(|root| ids[*root as usize]));
    let runner = runner.run(rules.iter());

    let mut result = WaferEGraphResult::status(STATUS_UNCHANGED);
    result.statistics.input_nodes = request.node_count;
    result.statistics.input_relations = request.relation_count;
    result.statistics.input_records = request
        .node_count
        .saturating_add(request.child_count)
        .saturating_add(request.relation_count)
        .saturating_add(request.root_count);
    result.statistics.input_bytes = request
        .node_count
        .saturating_mul(std::mem::size_of::<WaferEGraphNode>() as u64)
        .saturating_add(
            request
                .child_count
                .saturating_add(request.relation_count)
                .saturating_add(request.root_count)
                .saturating_mul(std::mem::size_of::<u32>() as u64),
        );
    result.statistics.relation_queries = runtime.relation_queries.load(AtomicOrdering::Relaxed);
    result.statistics.e_nodes = runner.egraph.total_number_of_nodes() as u64;
    result.statistics.e_classes = runner.egraph.number_of_classes() as u64;
    result.statistics.rewrite_matches = counters.matches.get();
    result.statistics.application_checks = runtime.application_checks.load(AtomicOrdering::Relaxed);
    result.statistics.unchanged_applications =
        runtime.unchanged_applications.load(AtomicOrdering::Relaxed);
    result.statistics.e_class_merges = counters.merges.get();
    result.statistics.rebuild_work = runner
        .iterations
        .iter()
        .map(|iteration| iteration.n_rebuilds as u64)
        .sum();
    result.statistics.iterations = runner.iterations.len() as u64;
    result.statistics.input_compute_occurrences = original_cost.compute_occurrences;
    result.statistics.input_access_occurrences = original_cost.access_occurrences;
    result.statistics.input_concat_occurrences = original_cost.concat_occurrences;
    result.statistics.identity_applications =
        runtime.identity_applications.load(AtomicOrdering::Relaxed);
    result.statistics.composition_applications = runtime
        .composition_applications
        .load(AtomicOrdering::Relaxed);
    result.statistics.compute_absorption_applications = runtime
        .compute_absorption_applications
        .load(AtomicOrdering::Relaxed);
    result.statistics.concat_applications =
        runtime.concat_applications.load(AtomicOrdering::Relaxed);
    result.statistics.result_reindex_applications = runtime
        .result_reindex_applications
        .load(AtomicOrdering::Relaxed);
    result.statistics.reshape_through_compute_applications = runtime
        .reshape_through_compute_applications
        .load(AtomicOrdering::Relaxed);

    let stop_reason = runner.stop_reason.as_ref().ok_or(STATUS_INTERNAL_ERROR)?;
    let search_limit_reached = counters.exhausted.get() || budget_stop(stop_reason);
    result.statistics.search_limit_reached = u64::from(search_limit_reached);
    if runtime.status() == RUNTIME_INTERNAL_ERROR
        || runner.egraph.classes().any(|class| !class.data.valid)
    {
        return Err(STATUS_INTERNAL_ERROR);
    }
    if runtime.status() == RUNTIME_BUDGET {
        result.status = STATUS_BUDGET_EXHAUSTED;
        return Ok(result);
    }
    if !search_limit_reached && !matches!(stop_reason, StopReason::Saturated) {
        return Err(STATUS_INTERNAL_ERROR);
    }

    // Runner rebuilds after applying rewrites even when a search limit stops
    // saturation. Those proven equalities remain valid: use the same bounded
    // extractor and complete DAG checks instead of discarding all progress.
    // A relation-service limit still prevents publishing an unverified result.
    let extraction_calls = Rc::new(Cell::new(0));
    let extractor = Extractor::new(
        &runner.egraph,
        GraphCostFunction {
            service,
            runtime: runtime.clone(),
            calls: extraction_calls.clone(),
        },
    );
    let mut selected_nodes: Vec<GraphLanguage> = Vec::new();
    let mut selected_node_ids: BTreeMap<GraphLanguage, u32> = BTreeMap::new();
    let mut output_roots = Vec::with_capacity(runner.roots.len());
    for root in &runner.roots {
        let (_root_cost, expression) = extractor.find_best(*root);
        let mut local_to_output = Vec::with_capacity(expression.as_ref().len());
        for node in expression.as_ref() {
            let mut canonical = node.clone();
            let mut canonical_children = Vec::with_capacity(node.children.len());
            for child in node.children.iter().copied() {
                let local = usize::from(child);
                let output = *local_to_output.get(local).ok_or(STATUS_INTERNAL_ERROR)?;
                canonical_children.push(Id::from(output as usize));
            }
            canonical.children = canonical_children.into_boxed_slice();
            let output = if let Some(existing) = selected_node_ids.get(&canonical) {
                *existing
            } else {
                let index =
                    u32::try_from(selected_nodes.len()).map_err(|_| STATUS_INTERNAL_ERROR)?;
                selected_node_ids.insert(canonical.clone(), index);
                selected_nodes.push(canonical);
                index
            };
            local_to_output.push(output);
        }
        output_roots.push(*local_to_output.last().ok_or(STATUS_INTERNAL_ERROR)?);
    }

    // The extractor's RecExpr order is topological but does not preserve the
    // source order of independent pure operations.  Re-establish the semantic
    // source-order tie-break while retaining every def-before-use edge.
    let mut dependents = vec![Vec::<usize>::new(); selected_nodes.len()];
    let mut remaining_children = Vec::with_capacity(selected_nodes.len());
    for (parent, node) in selected_nodes.iter().enumerate() {
        remaining_children.push(node.children.len());
        for child in node.children.iter().copied() {
            let child = usize::from(child);
            if child >= selected_nodes.len() {
                return Err(STATUS_INTERNAL_ERROR);
            }
            dependents[child].push(parent);
        }
    }
    let mut ready = BTreeSet::new();
    for (index, node) in selected_nodes.iter().enumerate() {
        if remaining_children[index] == 0 {
            ready.insert((node.source_order, node.kind, node.semantic_id, index));
        }
    }
    let mut order = Vec::with_capacity(selected_nodes.len());
    while let Some(key) = ready.pop_first() {
        let index = key.3;
        order.push(index);
        for dependent in &dependents[index] {
            remaining_children[*dependent] = remaining_children[*dependent]
                .checked_sub(1)
                .ok_or(STATUS_INTERNAL_ERROR)?;
            if remaining_children[*dependent] == 0 {
                let node = &selected_nodes[*dependent];
                ready.insert((node.source_order, node.kind, node.semantic_id, *dependent));
            }
        }
    }
    if order.len() != selected_nodes.len() {
        return Err(STATUS_INTERNAL_ERROR);
    }
    let mut old_to_new = vec![u32::MAX; selected_nodes.len()];
    for (new_index, old_index) in order.iter().copied().enumerate() {
        old_to_new[old_index] = u32::try_from(new_index).map_err(|_| STATUS_INTERNAL_ERROR)?;
    }
    let mut reordered = Vec::with_capacity(selected_nodes.len());
    for old_index in order {
        let mut node = selected_nodes[old_index].clone();
        node.children = node
            .children
            .iter()
            .map(|child| old_to_new[usize::from(*child)])
            .map(|child| Id::from(child as usize))
            .collect::<Vec<_>>()
            .into_boxed_slice();
        reordered.push(node);
    }
    for root in &mut output_roots {
        *root = old_to_new[*root as usize];
    }
    selected_nodes = reordered;

    result.statistics.extraction_work = extraction_calls.get();
    result.statistics.relation_queries = runtime.relation_queries.load(AtomicOrdering::Relaxed);
    if runtime.status() == RUNTIME_BUDGET {
        result.status = STATUS_BUDGET_EXHAUSTED;
        return Ok(result);
    }
    if !runtime.is_running() {
        return Err(STATUS_INTERNAL_ERROR);
    }
    let mut output_nodes = Vec::with_capacity(selected_nodes.len());
    let mut output_children = Vec::new();
    let mut output_relations = Vec::new();
    for node in &selected_nodes {
        let child_offset =
            u32::try_from(output_children.len()).map_err(|_| STATUS_INTERNAL_ERROR)?;
        let relation_offset =
            u32::try_from(output_relations.len()).map_err(|_| STATUS_INTERNAL_ERROR)?;
        for child in node.children.iter().copied() {
            output_children
                .push(u32::try_from(usize::from(child)).map_err(|_| STATUS_INTERNAL_ERROR)?);
        }
        output_relations.extend(node.relations.iter().copied());
        output_nodes.push(WaferEGraphNode {
            kind: node.kind,
            semantic_id: node.semantic_id,
            type_id: node.type_id,
            child_offset,
            child_count: u32::try_from(node.children.len()).map_err(|_| STATUS_INTERNAL_ERROR)?,
            relation_offset,
            relation_count: u32::try_from(node.relations.len())
                .map_err(|_| STATUS_INTERNAL_ERROR)?,
            data_input_count: node.data_input_count,
            axis: node.axis,
            source_order: node.source_order,
        });
    }
    let output_cost = raw_graph_cost(
        &output_roots,
        &output_nodes,
        &output_children,
        &output_relations,
        service,
        &runtime,
    )?;
    if !is_strictly_better(&output_cost, &original_cost) {
        return Ok(result);
    }

    let node_box = output_nodes.into_boxed_slice();
    let child_box = output_children.into_boxed_slice();
    let relation_box = output_relations.into_boxed_slice();
    let root_box = output_roots.into_boxed_slice();
    result.status = STATUS_CHANGED;
    result.node_count = node_box.len() as u64;
    result.child_count = child_box.len() as u64;
    result.relation_count = relation_box.len() as u64;
    result.root_count = root_box.len() as u64;
    result.statistics.output_nodes = result.node_count;
    result.statistics.output_records = result
        .node_count
        .saturating_add(result.child_count)
        .saturating_add(result.relation_count)
        .saturating_add(result.root_count);
    result.statistics.output_bytes = result
        .node_count
        .saturating_mul(std::mem::size_of::<WaferEGraphNode>() as u64)
        .saturating_add(
            result
                .child_count
                .saturating_add(result.relation_count)
                .saturating_add(result.root_count)
                .saturating_mul(std::mem::size_of::<u32>() as u64),
        );
    result.statistics.output_compute_occurrences = output_cost.compute_occurrences;
    result.statistics.output_access_occurrences = output_cost.access_occurrences;
    result.statistics.output_concat_occurrences = output_cost.concat_occurrences;
    result.nodes = Box::into_raw(node_box) as *mut WaferEGraphNode;
    result.children = Box::into_raw(child_box) as *mut u32;
    result.relations = Box::into_raw(relation_box) as *mut u32;
    result.root_nodes = Box::into_raw(root_box) as *mut u32;
    Ok(result)
}

#[no_mangle]
pub unsafe extern "C" fn waferRunStructuredEGraph(
    request: *const WaferEGraphRequest,
) -> *mut WaferEGraphResult {
    if request.is_null() {
        return Box::into_raw(Box::new(WaferEGraphResult::status(STATUS_INVALID_INPUT)));
    }
    let outcome = catch_unwind(AssertUnwindSafe(|| run(&*request)));
    let result = match outcome {
        Ok(Ok(result)) => result,
        Ok(Err(status)) => WaferEGraphResult::status(status),
        Err(_payload) => WaferEGraphResult::status(STATUS_INTERNAL_ERROR),
    };
    Box::into_raw(Box::new(result))
}

#[no_mangle]
pub unsafe extern "C" fn waferFreeStructuredEGraphResult(result: *mut WaferEGraphResult) {
    if result.is_null() {
        return;
    }
    let result = Box::from_raw(result);
    if !result.nodes.is_null() {
        let length = usize::try_from(result.node_count).unwrap_or(0);
        let slice = std::ptr::slice_from_raw_parts_mut(result.nodes, length);
        drop(Box::from_raw(slice));
    }
    if !result.children.is_null() {
        let length = usize::try_from(result.child_count).unwrap_or(0);
        let slice = std::ptr::slice_from_raw_parts_mut(result.children, length);
        drop(Box::from_raw(slice));
    }
    if !result.relations.is_null() {
        let length = usize::try_from(result.relation_count).unwrap_or(0);
        let slice = std::ptr::slice_from_raw_parts_mut(result.relations, length);
        drop(Box::from_raw(slice));
    }
    if !result.root_nodes.is_null() {
        let length = usize::try_from(result.root_count).unwrap_or(0);
        let slice = std::ptr::slice_from_raw_parts_mut(result.root_nodes, length);
        drop(Box::from_raw(slice));
    }
}

const _: () = {
    assert!(std::mem::size_of::<WaferEGraphNode>() == 40);
    assert!(std::mem::size_of::<WaferEGraphRelationFacts>() == 16);
};
