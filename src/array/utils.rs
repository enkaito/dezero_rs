use super::Array;

impl Array {
    pub fn all_close(&self, rhs: &Array, tol: f32) -> bool {
        (self - rhs)
            .data
            .iter()
            .try_for_each(|x| if x < &tol { Some(()) } else { None })
            .is_some()
    }

    pub fn size(&self) -> usize {
        self.size
    }
}
